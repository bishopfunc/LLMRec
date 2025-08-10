"""ユーザ履歴 + 候補アイテムから好み(正例)/非好み(負例)サンプルを LLM に問い合わせるパイプライン。

リファクタ方針:
 - グローバル処理を排除しクラス/関数へ責務分割
 - 冗長な再帰リトライを上限付き while ループへ置換
 - 型ヒント / Docstring を全面追加
 - I/O / LLM 要求 / プロンプト生成 / サンプル保存を疎結合化

想定入出力ファイル (work_dir 配下):
 - candidate_indices : pickle  (shape: [num_user, K])
 - train_mat         : pickle  (各ユーザの疎行列またはテンソル list/array)
 - item_attribute.csv: CSV (id,title,genre) あるいは (id,year,title) など
 - augmented_sample_dict: 途中結果 (user_id -> {0:pos,1:neg})

利用方法 (例):
    python gpt_ui_aug.py --work_dir ./data --model gpt-3.5-turbo --start 0 --end 100

環境変数:
 - OPENAI_API_KEY  (Baidu / OpenAI 互換エンドポイントのトークン)
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.llm import LLMClient


# ============================================================
# データアクセス
# ============================================================
@dataclass
class DataRepository:
    """学習に必要な各ファイルの読み書きを担当。"""

    work_dir: str
    candidate_file: str = "candidate_indices"
    train_mat_file: str = "train_mat"
    item_attribute_file: str = "item_attribute.csv"
    augmented_file: str = "augmented_sample_dict"

    # ---------- 読み込み ----------
    def load_candidates(self) -> Dict[int, Sequence[int]]:
        path = os.path.join(self.work_dir, self.candidate_file)
        arr = pickle.load(open(path, "rb"))  # shape (U, K)
        return {i: arr[i] for i in range(arr.shape[0])}

    def load_train_mat(self) -> Dict[int, Sequence[int]]:
        """train_mat からユーザ毎の履歴アイテム index 群を抽出。"""
        path = os.path.join(self.work_dir, self.train_mat_file)
        train_mat = pickle.load(open(path, "rb"))
        history: Dict[int, Sequence[int]] = {}
        for u in range(train_mat.shape[0]):
            _, cols = train_mat[u].nonzero()
            history[u] = cols
        return history

    def load_item_attribute(self) -> pd.DataFrame:
        path = os.path.join(self.work_dir, self.item_attribute_file)
        # 柔軟に列数対応: 3列 (id,title,genre) / 3列 (id,year,title) / 2列など
        df = pd.read_csv(path, header=None)
        # 列名推論
        if df.shape[1] == 3:
            # year が数値かどうかで分岐
            if pd.api.types.is_integer_dtype(df[1]) or pd.api.types.is_float_dtype(
                df[1]
            ):
                df.columns = ["id", "year", "title"]
            else:
                df.columns = ["id", "title", "genre"]
        elif df.shape[1] >= 2:
            df.columns = ["id", "title"] + [f"col{i}" for i in range(2, df.shape[1])]
        else:
            raise ValueError("item_attribute.csv の列数が不足しています")
        return df

    # ---------- augmented サンプル ----------
    def load_augmented(self) -> Dict[int, Dict[int, int]]:
        path = os.path.join(self.work_dir, self.augmented_file)
        if os.path.exists(path):
            try:
                return pickle.load(open(path, "rb"))
            except Exception as e:  # 破損時は新規
                print(f"[WARN] augmented_sample_dict load failed: {e}; recreate empty.")
        pickle.dump({}, open(path, "wb"))
        return {}

    def save_augmented(self, mapping: Dict[int, Dict[int, int]]) -> None:
        path = os.path.join(self.work_dir, self.augmented_file)
        pickle.dump(mapping, open(path, "wb"))


# ============================================================
# プロンプト生成
# ============================================================
class PromptBuilder:
    """ユーザ履歴 + 候補集合からお気に入り/非お気に入りを 2 本抽出させる指示文を構築。"""

    OUTPUT_SPEC = (
        "Please output the index of user's favorite and least favorite movie only from candidate, "
        "but not user history. Please get the index from candidate, at the beginning of each line.\n"
        "Output format:\nTwo numbers separated by '::'. Nothing else. "
        "Please just give the index of candidates (digits only, no brackets), and no reasoning.\n\n"
    )

    @staticmethod
    def build(
        item_df: pd.DataFrame, history: Sequence[int], candidates: Sequence[int]
    ) -> str:
        lines_hist = ["User history:"]
        use_year = "year" in item_df.columns
        for idx in history:
            try:
                title = item_df.loc[idx, "title"]
                if use_year:
                    year = item_df.loc[idx, "year"]
                    lines_hist.append(f"[{idx}] {year}, {title}")
                else:
                    lines_hist.append(f"[{idx}] {title}")
            except Exception:
                continue
        lines_cand = ["Candidates:"]
        for c in candidates:
            cid = c.item() if hasattr(c, "item") else int(c)
            if cid in item_df.index:
                title = item_df.loc[cid, "title"]
                if use_year:
                    year = item_df.loc[cid, "year"]
                    lines_cand.append(f"[{cid}] {year}, {title}")
                else:
                    lines_cand.append(f"[{cid}] {title}")
        return "\n".join(lines_hist + lines_cand) + "\n" + PromptBuilder.OUTPUT_SPEC


# ============================================================
# サンプル抽出パイプライン
# ============================================================
class PreferenceSampler:
    """ユーザ毎に LLM から (pos, neg) を抽出し永続化。"""

    def __init__(
        self,
        repo: DataRepository,
        client: LLMClient,
        item_df: pd.DataFrame,
        history: Dict[int, Sequence[int]],
        candidates: Dict[int, Sequence[int]],
        augmented: Dict[int, Dict[int, int]],
    ) -> None:
        self.repo = repo
        self.client = client
        self.item_df = item_df
        self.history = history
        self.candidates = candidates
        self.augmented = augmented  # user -> {0:pos,1:neg}

    @staticmethod
    def _parse_content(content: str) -> Optional[Tuple[int, int]]:
        parts = [p.strip() for p in content.replace("\n", "").split("::") if p.strip()]
        if len(parts) < 2:
            return None
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            return None

    def process_user(self, user_id: int) -> bool:
        """単一ユーザを処理。既に存在すればスキップ。

        戻り値: 新規取得=True / スキップ=False
        """
        if user_id in self.augmented:
            return False
        hist = self.history.get(user_id, [])
        cand = self.candidates.get(user_id, [])
        prompt = PromptBuilder.build(self.item_df, hist, cand)
        content = self.client.sample(prompt)
        print(f"[User {user_id}] Prompt:\n{prompt}\nResponse:\n{content}")
        if content is None:
            return False
        parsed = self._parse_content(content)
        if not parsed:
            print(f"[WARN] parse failed for user {user_id}: {content}")
            return False
        pos, neg = parsed
        self.augmented[user_id] = {0: pos, 1: neg}
        return True

    def run(
        self, start: int = 0, end: Optional[int] = None, save_interval: int = 50
    ) -> None:
        end = end if end is not None else max(self.history.keys()) + 1
        processed = 0
        for u in range(start, end):
            new_flag = self.process_user(u)
            if new_flag:
                processed += 1
            if (u + 1) % save_interval == 0:
                self.repo.save_augmented(self.augmented)
                print(f"[Progress] user {u + 1} saved (new={processed})")
        # final save
        self.repo.save_augmented(self.augmented)
        print(f"[Done] total new={processed}, total users stored={len(self.augmented)}")


# ============================================================
# エントリポイント
# ============================================================
def run_pipeline(
    work_dir: str,
    model: str = "gpt-3.5-turbo",
    start: int = 0,
    end: Optional[int] = None,
    endpoint: Optional[str] = None,
) -> None:
    """サンプル抽出パイプラインを実行。

    endpoint: None の場合はデフォルト (LLMConfig.endpoint)
    """

    repo = DataRepository(work_dir)
    item_df = repo.load_item_attribute()
    history = repo.load_train_mat()
    candidates = repo.load_candidates()
    augmented = repo.load_augmented()

    client = LLMClient()

    sampler = PreferenceSampler(repo, client, item_df, history, candidates, augmented)
    sampler.run(start=start, end=end)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Preference sampling via LLM")
    p.add_argument(
        "--work_dir",
        type=str,
        default="",
        help="作業ディレクトリ (データファイル群置き場)",
    )
    p.add_argument("--model", type=str, default="gpt-3.5-turbo", help="モデル名")
    p.add_argument("--endpoint", type=str, default=None, help="LLM エンドポイント URL")
    p.add_argument("--start", type=int, default=0, help="開始ユーザ index")
    p.add_argument(
        "--end", type=int, default=None, help="終了ユーザ index (Python range 末端)"
    )
    return p


if __name__ == "__main__":  # CLI 実行
    parser = _build_arg_parser()
    args = parser.parse_args()
    run_pipeline(
        work_dir=args.work_dir,
        model=args.model,
        start=args.start,
        end=args.end,
        endpoint=args.endpoint,
    )
