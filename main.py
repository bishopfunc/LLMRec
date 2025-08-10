"""学習エントリポイント (リファクタ版)

主な変更点:
 - 冗長インポート/重複コード削除
 - Trainer 内部ロジックをヘルパーメソッドに分割
 - 型ヒント / Docstring 付与
 - 可読性向上のための小規模関数抽出

既存挙動 (学習ループ / 損失計算 / early stopping) は原則維持。
"""

from __future__ import annotations

import math
import os
import pickle
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from time import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from Models import Decoder, MM_Model
from utility.batch_test import data_generator, test_torch  # 明示インポート
from utility.logging import Logger
from utility.parser import parse_args

args = parse_args()  # コマンドライン引数取得

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    """学習ループと評価を管理するクラス。

    責務:
      - データ/埋め込みロード
      - グラフ前処理 (正規化 & Tensor 化)
      - モデル / オプティマイザ初期化
      - エポック学習ループ & 評価 & 早期終了
    """

    def __init__(self, data_config: Dict[str, int]):
        self.task_name = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_{args.dataset}_{args.cf_model}"
        self.logger = Logger(filename=self.task_name, is_debug=args.debug)
        self.logger.logging(f"PID: {os.getpid()}")
        self.logger.logging(str(args))

        # --- ハイパーパラメータ ---
        self.mess_dropout: List[float] = eval(args.mess_dropout)
        self.lr: float = args.lr
        self.emb_dim: int = args.embed_size
        self.batch_size: int = args.batch_size
        self.weight_size: List[int] = eval(args.weight_size)
        self.n_layers: int = len(self.weight_size)
        self.regs: List[float] = eval(args.regs)
        self.decay: float = self.regs[0]

        # --- 特徴量ロード ---
        self.image_feats = np.load(
            os.path.join(args.data_path, args.dataset, "image_feat.npy")
        )
        self.text_feats = np.load(
            os.path.join(args.data_path, args.dataset, "text_feat.npy")
        )
        self.image_feat_dim = self.image_feats.shape[-1]
        self.text_feat_dim = self.text_feats.shape[-1]

        # --- グラフ & 埋め込み ---
        self.ui_graph = pickle.load(
            open(os.path.join(args.data_path, args.dataset, "train_mat"), "rb")
        )
        self.user_init_embedding = self._load_user_init_embedding()
        self.item_attribute_embedding = self._load_item_attribute_embedding()

        self.n_users = self.ui_graph.shape[0]
        self.n_items = self.ui_graph.shape[1]
        self.iu_graph = self.ui_graph.T

        # --- グラフ正規化 & Tensor 化 ---
        self.ui_graph = self._to_torch_graph(self.ui_graph, mean_flag=True)
        self.iu_graph = self._to_torch_graph(self.iu_graph, mean_flag=True)
        self.image_ui_graph = self.text_ui_graph = self.ui_graph
        self.image_iu_graph = self.text_iu_graph = self.iu_graph

        # --- モデル/オプティマイザ ---
        self.model_mm = MM_Model(
            self.n_users,
            self.n_items,
            self.emb_dim,
            self.weight_size,
            self.mess_dropout,
            self.image_feats,
            self.text_feats,
            self.user_init_embedding,
            self.item_attribute_embedding,
        ).to(device)
        self.decoder = Decoder(self.user_init_embedding.shape[1]).to(device)

        self.optimizer = optim.AdamW(
            [
                {"params": self.model_mm.parameters()},
            ],
            lr=self.lr,
        )
        self.de_optimizer = optim.AdamW(
            [
                {"params": self.decoder.parameters()},
            ],
            lr=args.de_lr,
        )

        # --- LLM 生成サンプル (ユーザ毎 pos/neg) をキャッシュ ---
        aug_path = os.path.join(args.data_path, args.dataset, "augmented_sample_dict")
        if os.path.exists(aug_path):
            self.augmented_sample_dict = pickle.load(open(aug_path, "rb"))
        else:
            self.augmented_sample_dict = {}
        # 評価指標で用いる K 群を一度だけ評価
        self.Ks: List[int] = list(eval(args.Ks))

    # -------------------------------------------------
    # データクラス: 1バッチ計算結果を保持
    # -------------------------------------------------
    @dataclass
    class BatchLoss:
        batch_loss: torch.Tensor
        mf_loss: torch.Tensor
        emb_loss: torch.Tensor
        reg_loss: torch.Tensor
        mm_mf_loss: torch.Tensor
        aug_mf_loss: torch.Tensor
        feat_emb_loss: torch.Tensor
        att_re_loss: torch.Tensor

    # =====================================================
    # 損失 / 正則化ユーティリティ
    # =====================================================
    def prune_loss(self, pred: torch.Tensor, drop_rate: float) -> torch.Tensor:
        """log-sigmoid 後スコア列から上位をドロップし残り平均。"""
        ind_sorted = torch.argsort(pred.detach())
        loss_sorted = pred[ind_sorted]
        remember_rate = 1 - drop_rate
        num_remember = int(remember_rate * len(loss_sorted))
        return loss_sorted[:num_remember].mean()

    def feat_reg_loss_calculation(
        self,
        g_item_image: torch.Tensor,
        g_item_text: torch.Tensor,
        g_user_image: torch.Tensor,
        g_user_text: torch.Tensor,
    ) -> torch.Tensor:
        feat_reg = (
            0.5 * (g_item_image**2).sum()
            + 0.5 * (g_item_text**2).sum()
            + 0.5 * (g_user_image**2).sum()
            + 0.5 * (g_user_text**2).sum()
        )
        feat_reg = feat_reg / self.n_items
        return args.feat_reg_decay * feat_reg

    def mse_criterion(
        self, x: torch.Tensor, y: torch.Tensor, alpha: float = 3
    ) -> torch.Tensor:
        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        y = torch.nn.functional.normalize(y, p=2, dim=-1)
        return torch.nn.functional.mse_loss(x, y)

    def sce_criterion(
        self, x: torch.Tensor, y: torch.Tensor, alpha: float = 1
    ) -> torch.Tensor:
        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        y = torch.nn.functional.normalize(y, p=2, dim=-1)
        return (1 - (x * y).sum(dim=-1)).pow_(alpha).mean()

    def bpr_loss(
        self, users: torch.Tensor, pos_items: torch.Tensor, neg_items: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pos_scores = (users * pos_items).sum(dim=1)
        neg_scores = (users * neg_items).sum(dim=1)
        regularizer = (
            1.0 / (2 * (users**2).sum() + 1e-8)
            + 1.0 / (2 * (pos_items**2).sum() + 1e-8)
            + 1.0 / (2 * (neg_items**2).sum() + 1e-8)
        )
        regularizer = regularizer / self.batch_size
        maxi = torch.logsigmoid(pos_scores - neg_scores + 1e-8)
        mf_loss = -self.prune_loss(maxi, args.prune_loss_drop_rate)
        emb_loss = self.decay * regularizer
        reg_loss = torch.tensor(0.0, device=device)
        return mf_loss, emb_loss, reg_loss

    # =====================================================
    # ロード補助
    # =====================================================
    def _load_user_init_embedding(self) -> np.ndarray:
        """ユーザ初期埋め込みをロードして numpy 配列に整形。"""
        path = os.path.join(
            args.data_path, args.dataset, "augmented_user_init_embedding"
        )
        emb_raw = pickle.load(open(path, "rb"))
        # 既存コードはインデックス順アクセス; 列挙で numpy 化
        arr_list = [emb_raw[i] for i in range(len(emb_raw))]
        arr = np.array(arr_list)
        out_path = os.path.join(
            args.data_path, args.dataset, "augmented_user_init_embedding_final"
        )
        pickle.dump(arr, open(out_path, "wb"))

        # =====================================================
        # 分割された学習処理
        # =====================================================

    def _train_one_epoch(self, epoch: int):
        """1 エポック分の全バッチを学習し損失統計を返す。"""
        t_start = time()
        aggregate_loss = mf_agg = emb_agg = reg_agg = 0.0
        contrastive_loss = 0.0  # プレースホルダ
        n_batch = data_generator.n_train // args.batch_size + 1
        for _ in tqdm(range(n_batch)):
            batch_losses = self._run_batch()
            aggregate_loss += float(batch_losses.batch_loss)
            mf_agg += float(batch_losses.mf_loss)
            emb_agg += float(batch_losses.emb_loss)
            reg_agg += float(batch_losses.reg_loss)

        if math.isnan(aggregate_loss):
            self.logger.logging("ERROR: loss is nan.")
            sys.exit()

        if (epoch + 1) % args.verbose != 0:  # 簡易ログ
            self._log_epoch_train(
                epoch,
                time() - t_start,
                aggregate_loss,
                mf_agg,
                emb_agg,
                reg_agg,
                contrastive_loss,
            )
        return dict(
            aggregate_loss=aggregate_loss,
            mf_agg=mf_agg,
            emb_agg=emb_agg,
            reg_agg=reg_agg,
            contrastive_loss=contrastive_loss,
        )

    def _run_batch(self) -> "Trainer.BatchLoss":
        """単一バッチのサンプリング→forward→損失計算→backward を実行。"""
        self.model_mm.train()
        users, pos_items, neg_items = data_generator.sample()
        users, pos_items, neg_items = self._augment_batch(users, pos_items, neg_items)

        (
            user_presentation_h,
            item_presentation_h,
            image_i_feat,
            text_i_feat,
            image_u_feat,
            text_u_feat,
            user_prof_feat_pre,
            item_prof_feat_pre,
            user_prof_feat,
            item_prof_feat,
            user_att_feats,
            item_att_feats,
            i_mask_nodes,
            u_mask_nodes,
        ) = self._forward_graph()

        # --- BPR 基本 ---
        u_bpr_emb = user_presentation_h[users]
        i_bpr_pos_emb = item_presentation_h[pos_items]
        i_bpr_neg_emb = item_presentation_h[neg_items]
        batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(
            u_bpr_emb, i_bpr_pos_emb, i_bpr_neg_emb
        )

        # --- マルチモーダル BPR ---
        image_batch_mf_loss, _, _ = self.bpr_loss(
            image_u_feat[users], image_i_feat[pos_items], image_i_feat[neg_items]
        )
        text_batch_mf_loss, _, _ = self.bpr_loss(
            text_u_feat[users], text_i_feat[pos_items], text_i_feat[neg_items]
        )
        mm_mf_loss = image_batch_mf_loss + text_batch_mf_loss

        # --- 属性拡張 BPR ---
        batch_mf_loss_aug = self._compute_attribute_augmented_bpr(
            users, pos_items, neg_items, user_prof_feat, item_att_feats
        )

        # --- 正則化/再構成 ---
        feat_emb_loss = self.feat_reg_loss_calculation(
            image_i_feat, text_i_feat, image_u_feat, text_u_feat
        )
        att_re_loss = self._attribute_reconstruction_loss(
            user_prof_feat, item_att_feats, u_mask_nodes, i_mask_nodes
        )

        batch_loss = (
            batch_mf_loss
            + batch_emb_loss
            + batch_reg_loss
            + feat_emb_loss
            + args.aug_mf_rate * batch_mf_loss_aug
            + args.mm_mf_rate * mm_mf_loss
            + args.att_re_rate * att_re_loss
        )

        # --- 最適化 ---
        nn.utils.clip_grad_norm_(self.model_mm.parameters(), max_norm=1.0)
        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()

        return Trainer.BatchLoss(
            batch_loss=batch_loss,
            mf_loss=batch_mf_loss,
            emb_loss=batch_emb_loss,
            reg_loss=batch_reg_loss,
            mm_mf_loss=mm_mf_loss,
            aug_mf_loss=batch_mf_loss_aug,
            feat_emb_loss=feat_emb_loss,
            att_re_loss=att_re_loss,
        )

    def _maybe_eval_and_update(
        self,
        ret: Dict[str, List[float]],
        users_to_test: List[int],
        best_recall: float,
        test_ret: Optional[Dict[str, List[float]]],
        stopping_step: int,
    ) -> Tuple[bool, float, Optional[Dict[str, List[float]]], int]:
        """ベスト更新または EarlyStopping カウンタ更新。

        Returns:
            improved: ベスト更新があったか
            best_recall: 更新後のベスト値
            test_ret: ベスト時のテスト結果
            stopping_step: 更新される early stop カウンタ
        """
        if ret["recall"][1] > best_recall:
            best_recall = ret["recall"][1]
            test_ret = self.test(users_to_test, is_val=False)
            self.logger.logging(
                "Test_Recall@%d: %.5f,  precision=[%.5f], ndcg=[%.5f]"
                % (
                    self.Ks[1],
                    test_ret["recall"][1],
                    test_ret["precision"][1],
                    test_ret["ndcg"][1],
                )
            )
            stopping_step = 0
            return True, best_recall, test_ret, stopping_step
        else:
            if stopping_step < args.early_stopping_patience:
                stopping_step += 1
                self.logger.logging(f"#####Early stopping steps: {stopping_step} #####")
            return False, best_recall, test_ret, stopping_step

    def _log_full_metrics(
        self,
        epoch: int,
        epoch_metrics: Dict[str, float],
        ret: Dict[str, List[float]],
    ) -> None:
        """詳細指標ログ (verbose>0)。"""
        self.logger.logging(
            (
                "Epoch %d: train==[%.5f=%.5f + %.5f + %.5f], recall=[%.5f, %.5f, %.5f, %.5f], precision=[%.5f, %.5f, %.5f, %.5f], hit=[%.5f, %.5f, %.5f, %.5f], ndcg=[%.5f, %.5f, %.5f, %.5f]"
            )
            % (
                epoch,
                epoch_metrics["aggregate_loss"],
                epoch_metrics["mf_agg"],
                epoch_metrics["emb_agg"],
                epoch_metrics["reg_agg"],
                ret["recall"][0],
                ret["recall"][1],
                ret["recall"][2],
                ret["recall"][-1],
                ret["precision"][0],
                ret["precision"][1],
                ret["precision"][2],
                ret["precision"][-1],
                ret["hit_ratio"][0],
                ret["hit_ratio"][1],
                ret["hit_ratio"][2],
                ret["hit_ratio"][-1],
                ret["ndcg"][0],
                ret["ndcg"][1],
                ret["ndcg"][2],
                ret["ndcg"][-1],
            )
        )

    def _compute_attribute_augmented_bpr(
        self,
        users: List[int],
        pos_items: List[int],
        neg_items: List[int],
        user_prof_feat: torch.Tensor,
        item_att_feats: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """属性ごとの埋め込みを使った BPR を合算。"""
        total = torch.tensor(0.0, device=device)
        if not item_att_feats:
            return total
        for tensor in item_att_feats.values():
            u_emb_aug = user_prof_feat[users]
            pos_emb_aug = tensor[pos_items]
            neg_emb_aug = tensor[neg_items]
            tmp_mf, _, _ = self.bpr_loss(u_emb_aug, pos_emb_aug, neg_emb_aug)
            total += tmp_mf
        return total

    def _log_epoch_train(
        self,
        epoch: int,
        t_elapsed: float,
        aggregate_loss: float,
        mf_agg: float,
        emb_agg: float,
        reg_agg: float,
        contrastive_loss: float,
    ) -> None:
        """簡易 train ログ (verbose 間引き用)。"""
        perf_str = f"Epoch {epoch} [{t_elapsed:.1f}s]: train==[{aggregate_loss:.5f}={mf_agg:.5f} + {emb_agg:.5f} + {reg_agg:.5f}  + {contrastive_loss:.5f}]"
        self.logger.logging(perf_str)

    # =====================================================
    # 評価 / 学習
    # =====================================================
    def test(
        self, users_to_test: Sequence[int], is_val: bool
    ) -> Dict[str, List[float]]:
        self.model_mm.eval()
        with torch.no_grad():
            ua_embeddings, ia_embeddings, *rest = self.model_mm(
                self.ui_graph,
                self.iu_graph,
                self.image_ui_graph,
                self.image_iu_graph,
                self.text_ui_graph,
                self.text_iu_graph,
            )
        return test_torch(ua_embeddings, ia_embeddings, users_to_test, is_val)

    def _attribute_reconstruction_loss(
        self,
        user_prof_feat: torch.Tensor,
        item_att_feats: Dict[str, torch.Tensor],
        u_mask_nodes: Sequence[int],
        i_mask_nodes: Sequence[int],
    ) -> torch.Tensor:
        if not args.mask:
            return torch.tensor(0.0, device=device)
        input_i = {k: v[i_mask_nodes] for k, v in item_att_feats.items()}
        decoded_u, decoded_i = self.decoder(
            torch.tensor(user_prof_feat[u_mask_nodes]).to(device), input_i
        )
        loss = torch.tensor(0.0, device=device)
        criterion = (
            self.mse_criterion if args.feat_loss_type == "mse" else self.sce_criterion
        )
        loss = loss + criterion(
            decoded_u,
            torch.tensor(self.user_init_embedding[u_mask_nodes]).to(device),
            alpha=args.alpha_l,
        )
        for idx, k in enumerate(item_att_feats.keys()):
            target = torch.tensor(self.item_attribute_embedding[k][i_mask_nodes]).to(
                device
            )
            loss = loss + criterion(decoded_i[idx], target, alpha=args.alpha_l)
        return loss

    def train(self) -> Tuple[float, str]:
        now_time = datetime.now()
        run_time = datetime.strftime(now_time, "%Y_%m_%d__%H_%M_%S")
        training_time_list: List[float] = []
        stopping_step = 0
        best_recall = 0.0
        test_ret = None  # best モデル評価保持

        for epoch in range(args.epoch):
            t1 = time()
            aggregate_loss = mf_agg = emb_agg = reg_agg = 0.0
            contrastive_loss = 0.0  # プレースホルダ
            n_batch = data_generator.n_train // args.batch_size + 1

            for _ in tqdm(range(n_batch)):
                self.model_mm.train()  # 1 バッチ学習開始
                users, pos_items, neg_items = data_generator.sample()  # サンプリング
                users, pos_items, neg_items = self._augment_batch(
                    users, pos_items, neg_items
                )  # LLM 拡張

                # Forward (グラフ伝播と特徴生成)
                (
                    user_presentation_h,
                    item_presentation_h,
                    image_i_feat,
                    text_i_feat,
                    image_u_feat,
                    text_u_feat,
                    user_prof_feat_pre,
                    item_prof_feat_pre,
                    user_prof_feat,
                    item_prof_feat,
                    user_att_feats,
                    item_att_feats,
                    i_mask_nodes,
                    u_mask_nodes,
                ) = self._forward_graph()

                # BPR 基本損失
                u_bpr_emb = user_presentation_h[users]
                i_bpr_pos_emb = item_presentation_h[pos_items]
                i_bpr_neg_emb = item_presentation_h[neg_items]
                batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(
                    u_bpr_emb, i_bpr_pos_emb, i_bpr_neg_emb
                )

                # マルチモーダル (image/text)
                image_batch_mf_loss, _, _ = self.bpr_loss(
                    image_u_feat[users],
                    image_i_feat[pos_items],
                    image_i_feat[neg_items],
                )
                text_batch_mf_loss, _, _ = self.bpr_loss(
                    text_u_feat[users], text_i_feat[pos_items], text_i_feat[neg_items]
                )
                mm_mf_loss = image_batch_mf_loss + text_batch_mf_loss

                # 属性別 BPR (各属性で BPR を計算し総和)
                batch_mf_loss_aug = self._compute_attribute_augmented_bpr(
                    users, pos_items, neg_items, user_prof_feat, item_att_feats
                )

                feat_emb_loss = self.feat_reg_loss_calculation(
                    image_i_feat, text_i_feat, image_u_feat, text_u_feat
                )
                att_re_loss = self._attribute_reconstruction_loss(
                    user_prof_feat, item_att_feats, u_mask_nodes, i_mask_nodes
                )

                batch_loss = (
                    batch_mf_loss
                    + batch_emb_loss
                    + batch_reg_loss
                    + feat_emb_loss
                    + args.aug_mf_rate * batch_mf_loss_aug
                    + args.mm_mf_rate * mm_mf_loss
                    + args.att_re_rate * att_re_loss
                )

                nn.utils.clip_grad_norm_(self.model_mm.parameters(), max_norm=1.0)
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

                aggregate_loss += float(batch_loss)
                mf_agg += float(batch_mf_loss)
                emb_agg += float(batch_emb_loss)
                reg_agg += float(batch_reg_loss)

            if math.isnan(aggregate_loss):
                self.logger.logging("ERROR: loss is nan.")
                sys.exit()

            if (epoch + 1) % args.verbose != 0:  # 詳細ログを間引く
                training_time_list.append(time() - t1)
                self._log_epoch_train(
                    epoch,
                    time() - t1,
                    aggregate_loss,
                    mf_agg,
                    emb_agg,
                    reg_agg,
                    contrastive_loss,
                )

            t2 = time()
            users_to_test = list(data_generator.test_set.keys())
            ret = self.test(users_to_test, is_val=False)
            training_time_list.append(t2 - t1)
            t3 = time()

            if args.verbose > 0:
                self.logger.logging(
                    (
                        "Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f], recall=[%.5f, %.5f, %.5f, %.5f], precision=[%.5f, %.5f, %.5f, %.5f], hit=[%.5f, %.5f, %.5f, %.5f], ndcg=[%.5f, %.5f, %.5f, %.5f]"
                    )
                    % (
                        epoch,
                        t2 - t1,
                        t3 - t2,
                        aggregate_loss,
                        mf_agg,
                        emb_agg,
                        reg_agg,
                        ret["recall"][0],
                        ret["recall"][1],
                        ret["recall"][2],
                        ret["recall"][-1],
                        ret["precision"][0],
                        ret["precision"][1],
                        ret["precision"][2],
                        ret["precision"][-1],
                        ret["hit_ratio"][0],
                        ret["hit_ratio"][1],
                        ret["hit_ratio"][2],
                        ret["hit_ratio"][-1],
                        ret["ndcg"][0],
                        ret["ndcg"][1],
                        ret["ndcg"][2],
                        ret["ndcg"][-1],
                    )
                )

            if ret["recall"][1] > best_recall:
                best_recall = ret["recall"][1]
                test_ret = self.test(users_to_test, is_val=False)
                self.logger.logging(
                    "Test_Recall@%d: %.5f,  precision=[%.5f], ndcg=[%.5f]"
                    % (
                        eval(args.Ks)[1],
                        test_ret["recall"][1],
                        test_ret["precision"][1],
                        test_ret["ndcg"][1],
                    )
                )
                stopping_step = 0
            elif stopping_step < args.early_stopping_patience:
                stopping_step += 1
                self.logger.logging(f"#####Early stopping steps: {stopping_step} #####")
            else:
                self.logger.logging("#####Early stop! #####")
                break

        if test_ret is not None:
            self.logger.logging(str(test_ret))  # ループ終了後のベスト結果
        else:
            self.logger.logging("No improvement observed; no test_ret captured.")
        return best_recall, run_time


def set_seed(seed: int) -> None:
    """乱数シード統一。"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":  # 実行エントリ
    set_seed(args.seed)
    config = {
        "n_users": data_generator.n_users,
        "n_items": data_generator.n_items,
    }
    trainer = Trainer(data_config=config)
    trainer.train()
