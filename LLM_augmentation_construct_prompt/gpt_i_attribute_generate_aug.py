"""映画アイテム属性の LLM 補完 + 埋め込み生成スクリプト（リファクタ版）

主な責務分割:
 1. PromptBuilder: プロンプト生成
 2. AttributeAugmenter: 監督 / 国 / 言語 属性の補完
 3. EmbeddingGenerator: 各属性テキストの埋め込み生成
 4. DataAggregator: 部分保存された埋め込み shard を統合
 5. run_pipeline: 一括オーケストレーション

特徴:
 - 冗長/重複していた LLM_request 関数群を統合
 - 再帰リトライを while+カウンタ型へ変更（コード簡素化）
 - Type Hint & 日本語 Docstring 整備
 - 可搬性向上（環境変数で API キー取得）
 - 埋め込み生成は shard 部分保存に対応
"""

from __future__ import annotations

import os
import pickle
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import requests

# torch は任意利用（埋め込みテンソル化が必要な場合のみ）
try:  # インポート失敗しても致命的ではない
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore


# =============================================================
# 設定データクラス
# =============================================================
@dataclass
class LLMConfig:
    """LLM / Embedding API 設定。

    api_key_env: 環境変数名 (例: OPENAI_API_KEY)
    attribute_model: 作品属性補完で使用するモデル識別子
    embedding_model: 埋め込み生成モデル識別子
    attribute_endpoint: Completions/Chat など属性取得用エンドポイント URL
    embedding_endpoint: Embedding 用エンドポイント URL
    max_retries: 失敗時リトライ回数
    retry_sleep_sec: リトライ間隔秒
    temperature: LLM 温度
    top_p: nucleus sampling
    timeout_sec: HTTP タイムアウト
    """

    api_key_env: str = "OPENAI_API_KEY"
    attribute_model: str = "text-davinci-003"
    embedding_model: str = "text-embedding-ada-002"
    attribute_endpoint: str = "https://api.openai.com/v1/completions"
    embedding_endpoint: str = "https://api.openai.com/v1/embeddings"
    max_retries: int = 5
    retry_sleep_sec: float = 5.0
    temperature: float = 0.6
    top_p: float = 1.0
    timeout_sec: float = 60.0

    def api_key(self) -> str:
        key = os.getenv(self.api_key_env, "")
        if not key:
            raise RuntimeError(
                f"環境変数 {self.api_key_env} が設定されていません (API キー未取得)"
            )
        return key


# =============================================================
# プロンプト生成
# =============================================================
class PromptBuilder:
    """映画アイテム属性補完用プロンプトを組み立てる。"""

    PREAMBLE = "You are now a search engines, and required to provide the inquired information of the given movies below:\n"
    OUTPUT_SPEC = (
        "The inquired information is : director, country, language.\n"
        "And please output them in form of: \n"
        "director::country::language\n"
        "please output only the content in the form above, i.e., director::country::language\n"
        ", but no other thing else, no reasoning, no index.\n\n"
    )

    @staticmethod
    def build(items: pd.DataFrame, indices: Sequence[int]) -> str:
        """指定 index の映画 (year, title) を列挙し属性問い合わせプロンプトを生成。"""
        lines: List[str] = []
        for idx in indices:
            year = items.loc[idx, "year"] if "year" in items.columns else ""
            title = items.loc[idx, "title"]
            prefix = f"[{idx}] "
            mid = f"{year}, {title}" if year != "" else title
            lines.append(prefix + mid)
        return (
            PromptBuilder.PREAMBLE + "\n".join(lines) + "\n" + PromptBuilder.OUTPUT_SPEC
        )


# =============================================================
# 属性補完
# =============================================================
class AttributeAugmenter:
    """LLM を用いて (director, country, language) を補完し pickle に保存。"""

    def __init__(self, cfg: LLMConfig, work_dir: str) -> None:
        self.cfg = cfg
        self.work_dir = work_dir
        self.cache_path = os.path.join(work_dir, "augmented_attribute_dict")
        self.attributes: Dict[int, Dict[int, str]] = self._load_cache()

    # ------------------------- persistence ---------------------
    def _load_cache(self) -> Dict[int, Dict[int, str]]:
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "rb") as f:
                    return pickle.load(f)
            except Exception:  # 破損時は空で再生成
                return {}
        return {}

    def _save_cache(self) -> None:
        with open(self.cache_path, "wb") as f:
            pickle.dump(self.attributes, f)

    # ------------------------- core ----------------------------
    def fetch(self, df: pd.DataFrame, indices: Sequence[int]) -> bool:
        """指定 index 群の属性取得。成功/既存なら True。失敗で False。

        LLM 応答例 (複数行対応):
            Director A::USA::English\nDirector B::France::French
        """
        # 既存チェック
        if all(i in self.attributes for i in indices):
            return True

        prompt = PromptBuilder.build(df, indices)
        headers = {"Authorization": f"Bearer {self.cfg.api_key()}"}
        payload = {
            "model": self.cfg.attribute_model,
            "prompt": prompt,
            "max_tokens": 512,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
        }
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                resp = requests.post(
                    self.cfg.attribute_endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.cfg.timeout_sec,
                )
                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["text"].strip()
                rows = [r for r in text.splitlines() if r.strip()]
                for idx, row in zip(indices, rows):
                    parts = row.split("::")
                    if len(parts) < 3:
                        continue  # フォーマット崩れはスキップ
                    director, country, language = parts[0], parts[1], parts[2]
                    self.attributes[idx] = {0: director, 1: country, 2: language}
                self._save_cache()
                return True
            except Exception as e:  # broad (通信/パース共通でリトライ)
                print(f"[AttributeAugmenter] attempt {attempt} failed: {e}")
                if attempt == self.cfg.max_retries:
                    return False
                time.sleep(self.cfg.retry_sleep_sec)
        return False

    def to_augmented_dataframe(self, base_filtered_csv: str) -> pd.DataFrame:
        """フィルタ済み item_attribute_filter.csv を拡張列付きで DataFrame 化。"""
        df = pd.read_csv(base_filtered_csv, names=["id", "year", "title"], header=None)
        directors, countries, languages = [], [], []
        for i in range(len(self.attributes)):
            attr = self.attributes.get(i, {})
            directors.append(attr.get(0, ""))
            countries.append(attr.get(1, ""))
            languages.append(attr.get(2, ""))
        df["director"] = pd.Series(directors)
        df["country"] = pd.Series(countries)
        df["language"] = pd.Series(languages)
        return df


# =============================================================
# 埋め込み生成
# =============================================================
class EmbeddingGenerator:
    """指定属性列テキストを Embedding API へ送りベクトルを辞書保存。"""

    def __init__(self, cfg: LLMConfig, work_dir: str, file_prefix: str = "embedding"):
        self.cfg = cfg
        self.work_dir = work_dir
        self.file_prefix = file_prefix

    def _save_partial(
        self, mapping: Dict[str, Dict[int, List[float]]], shard: int
    ) -> str:
        path = os.path.join(self.work_dir, f"{self.file_prefix}_part{shard}.pkl")
        with open(path, "wb") as f:
            pickle.dump(mapping, f)
        return path

    def generate(
        self,
        df: pd.DataFrame,
        attributes: Sequence[str],
        start: int = 0,
        end: Optional[int] = None,
        shard_interval: int = 1000,
    ) -> Dict[str, Dict[int, List[float]]]:
        """指定行範囲の属性テキストを埋め込み生成。

        shard_interval: その件数ごとに部分保存（再開容易化）
        """
        end = end if end is not None else len(df)
        store: Dict[str, Dict[int, List[float]]] = {a: {} for a in attributes}
        headers = {"Authorization": f"Bearer {self.cfg.api_key()}"}
        for i in range(start, end):
            for attr in attributes:
                if i in store[attr]:
                    continue
                text = str(df.loc[i, attr])
                payload = {"model": self.cfg.embedding_model, "input": text}
                success = False
                for attempt in range(1, self.cfg.max_retries + 1):
                    try:
                        r = requests.post(
                            self.cfg.embedding_endpoint,
                            headers=headers,
                            json=payload,
                            timeout=self.cfg.timeout_sec,
                        )
                        r.raise_for_status()
                        emb = r.json()["data"][0]["embedding"]
                        store[attr][i] = emb
                        success = True
                        break
                    except Exception as e:
                        print(
                            f"[EmbeddingGenerator] attr={attr} idx={i} attempt {attempt} failed: {e}"
                        )
                        if attempt == self.cfg.max_retries:
                            # 諦めて空配列を埋める（長さ不一致回避）
                            store[attr][i] = []
                        else:
                            time.sleep(self.cfg.retry_sleep_sec)
                if not success:
                    continue
            # shard 保存
            if (i + 1) % shard_interval == 0:
                self._save_partial(store, shard=(i + 1) // shard_interval)
        # 最終保存
        self._save_partial(store, shard=0)
        return store

    @staticmethod
    def to_matrix(mapping: Dict[str, Dict[int, List[float]]]) -> Dict[str, np.ndarray]:
        """dict[attr][idx] -> np.ndarray[attr]=(N, D) へ変換。不揃い/欠損はゼロ埋め。"""
        out: Dict[str, np.ndarray] = {}
        for attr, rows in mapping.items():
            if not rows:
                out[attr] = np.zeros((0, 0), dtype=float)
                continue
            max_index = max(rows.keys())
            # 埋め込み次元推測
            first_vec = next((v for v in rows.values() if v), [])
            dim = len(first_vec)
            mat = np.zeros((max_index + 1, dim), dtype=float)
            for idx, vec in rows.items():
                if len(vec) == dim:
                    mat[idx] = np.array(vec, dtype=float)
            out[attr] = mat
        return out


# =============================================================
# 集約 / パイプライン
# =============================================================
class DataAggregator:
    """分割保存された埋め込み shard を統合するユーティリティ。"""

    @staticmethod
    def merge_embedding_parts(
        work_dir: str, prefix: str = "embedding_part"
    ) -> Dict[str, Dict[int, List[float]]]:
        """prefix に一致する shard pkl を読み込み統合した mapping を返す。

        戻り値:
            attr -> { index -> embedding(list[float]) }
        """
        merged: Dict[str, Dict[int, List[float]]] = {}
        for name in sorted(os.listdir(work_dir)):
            if not name.startswith(prefix) or not name.endswith(".pkl"):
                continue
            path = os.path.join(work_dir, name)
            try:
                with open(path, "rb") as f:
                    part = pickle.load(f)
                for attr, mapping in part.items():
                    merged.setdefault(attr, {}).update(mapping)
            except Exception as e:  # 破損はスキップ
                print(f"[DataAggregator] skip {name}: {e}")
        return merged


def run_pipeline(
    work_dir: str,
    item_attribute_csv: str = "item_attribute.csv",
    item_attribute_filtered_csv: str = "item_attribute_filter.csv",
    output_augmented_csv: str = "augmented_item_attribute_agg.csv",
    start_index: int = 0,
    end_index: Optional[int] = None,
    generate_embeddings: bool = False,
    embedding_attrs: Optional[Sequence[str]] = None,
) -> None:
    """一連の処理を実行。

    generate_embeddings: True の場合、指定属性列の埋め込みも生成
    embedding_attrs: None の場合デフォルト (year,title,director,country,language)
    """

    cfg = LLMConfig()
    augmenter = AttributeAugmenter(cfg, work_dir)

    # --- 1. 属性補完 ---
    item_path = os.path.join(work_dir, item_attribute_csv)
    items = pd.read_csv(item_path, names=["id", "year", "title"], header=None)
    end_index = end_index if end_index is not None else len(items)

    for i in range(start_index, end_index):
        # 1件ずつ (必要ならバッチ拡張)
        augmenter.fetch(items, [i])
        if (i + 1) % 200 == 0:
            print(f"[AttributeAugmenter] processed {i + 1} items")

    # --- 2. 拡張 CSV 出力 ---
    filtered_path = os.path.join(work_dir, item_attribute_filtered_csv)
    augmented_df = augmenter.to_augmented_dataframe(filtered_path)
    augmented_csv_path = os.path.join(work_dir, output_augmented_csv)
    augmented_df.to_csv(augmented_csv_path, index=False, header=False)
    print(f"[Output] augmented csv -> {augmented_csv_path}")

    # --- 3. 埋め込み生成 (任意) ---
    if generate_embeddings:
        attrs = (
            list(embedding_attrs)
            if embedding_attrs is not None
            else ["year", "title", "director", "country", "language"]
        )
        emb_gen = EmbeddingGenerator(cfg, work_dir)
        emb_map = emb_gen.generate(augmented_df, attrs)
        matrices = emb_gen.to_matrix(emb_map)
        with open(os.path.join(work_dir, "embedding_matrices.pkl"), "wb") as f:
            pickle.dump(matrices, f)
        print("[Output] embeddings saved.")


# =============================================================
# CLI 実行サポート
# =============================================================
if __name__ == "__main__":
    WORK_DIR = os.getenv("LLMREC_WORKDIR", "")  # 空ならカレント
    START = int(os.getenv("LLMREC_START", "0"))
    END_ENV = os.getenv("LLMREC_END")
    END = int(END_ENV) if END_ENV else None
    DO_EMB = os.getenv("LLMREC_EMB", "0") == "1"
    run_pipeline(
        work_dir=WORK_DIR,
        start_index=START,
        end_index=END,
        generate_embeddings=DO_EMB,
    )
