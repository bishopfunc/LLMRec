import math
import os
import pickle
import random
import sys
from datetime import datetime
from time import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from Models import Decoder, MM_Model
from utility.batch_test import *
from utility.logging import Logger
from utility.parser import parse_args

args = parse_args()


class Trainer(object):
    def __init__(self, data_config):
        """Trainer初期化
        data_config: データ統計情報(n_users, n_items など)
        - ログ/パラメータ設定
        - 特徴量/拡張埋め込み/属性埋め込み読込
        - ユーザ-アイテム疎行列正規化 & Tensor化
        - モデル & オプティマイザ準備
        """
        self.task_name = "%s_%s_%s" % (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            args.dataset,
            args.cf_model,
        )
        self.logger = Logger(filename=self.task_name, is_debug=args.debug)
        self.logger.logging("PID: %d" % os.getpid())
        self.logger.logging(str(args))

        self.mess_dropout = eval(args.mess_dropout)
        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.weight_size = eval(args.weight_size)
        self.n_layers = len(self.weight_size)
        self.regs = eval(args.regs)
        self.decay = self.regs[0]

        # 画像/テキスト特徴量 (事前抽出済み) 読込
        self.image_feats = np.load(
            args.data_path + "{}/image_feat.npy".format(args.dataset)
        )
        self.text_feats = np.load(
            args.data_path + "{}/text_feat.npy".format(args.dataset)
        )
        self.image_feat_dim = self.image_feats.shape[-1]
        self.text_feat_dim = self.text_feats.shape[-1]

        # ユーザ×アイテム相互作用行列 (train)
        self.ui_graph = self.ui_graph_raw = pickle.load(
            open(args.data_path + args.dataset + "/train_mat", "rb")
        )
        # 拡張ユーザ初期埋め込み (LLM生成等) を numpy 配列化
        augmented_user_init_embedding = pickle.load(
            open(args.data_path + args.dataset + "/augmented_user_init_embedding", "rb")
        )
        augmented_user_init_embedding_list = []
        for i in range(len(augmented_user_init_embedding)):
            augmented_user_init_embedding_list.append(augmented_user_init_embedding[i])
        augmented_user_init_embedding_final = np.array(
            augmented_user_init_embedding_list
        )
        pickle.dump(
            augmented_user_init_embedding_final,
            open(
                args.data_path + args.dataset + "/augmented_user_init_embedding_final",
                "wb",
            ),
        )
        self.user_init_embedding = pickle.load(
            open(
                args.data_path + args.dataset + "/augmented_user_init_embedding_final",
                "rb",
            )
        )
        # 属性毎アイテム埋め込み辞書構築
        if args.dataset == "preprocessed_raw_MovieLens":
            augmented_total_embed_dict = {
                "title": [],
                "genre": [],
                "director": [],
                "country": [],
                "language": [],
            }
        elif args.dataset == "netflix_valid_item":
            augmented_total_embed_dict = {
                "year": [],
                "title": [],
                "director": [],
                "country": [],
                "language": [],
            }
        augmented_atttribute_embedding_dict = pickle.load(
            open(
                args.data_path + args.dataset + "/augmented_atttribute_embedding_dict",
                "rb",
            )
        )
        for value in augmented_atttribute_embedding_dict.keys():
            for i in range(len(augmented_atttribute_embedding_dict[value])):
                augmented_total_embed_dict[value].append(
                    augmented_atttribute_embedding_dict[value][i]
                )
            augmented_total_embed_dict[value] = np.array(
                augmented_total_embed_dict[value]
            )
        pickle.dump(
            augmented_total_embed_dict,
            open(args.data_path + args.dataset + "/augmented_total_embed_dict", "wb"),
        )
        self.item_attribute_embedding = pickle.load(
            open(args.data_path + args.dataset + "/augmented_total_embed_dict", "rb")
        )

        self.image_ui_index = {"x": [], "y": []}
        self.text_ui_index = {"x": [], "y": []}

        self.n_users = self.ui_graph.shape[0]
        self.n_items = self.ui_graph.shape[1]
        self.iu_graph = self.ui_graph.T

        # 行方向正規化 (mean_flag=True) -> Tensor へ変換
        self.ui_graph = self.csr_norm(self.ui_graph, mean_flag=True)
        self.iu_graph = self.csr_norm(self.iu_graph, mean_flag=True)
        self.ui_graph = self.matrix_to_tensor(self.ui_graph)
        self.iu_graph = self.matrix_to_tensor(self.iu_graph)
        self.image_ui_graph = self.text_ui_graph = self.ui_graph
        self.image_iu_graph = self.text_iu_graph = self.iu_graph

        # マルチモーダルモデル + 属性再構成デコーダ
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
        )
        self.model_mm = self.model_mm.cuda()
        self.decoder = Decoder(self.user_init_embedding.shape[1]).cuda()

        # Optimizer
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

    def csr_norm(self, csr_mat, mean_flag=False):
        """疎行列正規化
        mean_flag=False: D^-1/2 A D^-1/2 (対称)
        mean_flag=True : D^-1/2 A (行方向のみ)
        """
        rowsum = np.array(csr_mat.sum(1))
        rowsum = np.power(rowsum + 1e-8, -0.5).flatten()
        rowsum[np.isinf(rowsum)] = 0.0
        rowsum_diag = sp.diags(rowsum)
        colsum = np.array(csr_mat.sum(0))
        colsum = np.power(colsum + 1e-8, -0.5).flatten()
        colsum[np.isinf(colsum)] = 0.0
        colsum_diag = sp.diags(colsum)
        if mean_flag == False:
            return rowsum_diag * csr_mat * colsum_diag
        else:
            return rowsum_diag * csr_mat

    def matrix_to_tensor(self, cur_matrix):
        """scipy疎行列 -> GPU上 torch.sparse.FloatTensor"""
        if type(cur_matrix) != sp.coo_matrix:
            cur_matrix = cur_matrix.tocoo()  #
        indices = torch.from_numpy(
            np.vstack((cur_matrix.row, cur_matrix.col)).astype(np.int64)
        )  #
        values = torch.from_numpy(cur_matrix.data)  #
        shape = torch.Size(cur_matrix.shape)
        return (
            torch.sparse.FloatTensor(indices, values, shape).to(torch.float32).cuda()
        )  #

    def innerProduct(self, u_pos, i_pos, u_neg, j_neg):
        """(未使用ヘルパ) BPR用の内積計算"""
        pred_i = torch.sum(torch.mul(u_pos, i_pos), dim=-1)
        pred_j = torch.sum(torch.mul(u_neg, j_neg), dim=-1)
        return pred_i, pred_j

    def weights_init(self, m):
        """Linear層初期化 (Kaiming)"""
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0)

    def sim(self, z1, z2):
        """正規化後埋め込みのコサイン類似度(=内積)"""
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def feat_reg_loss_calculation(
        self, g_item_image, g_item_text, g_user_image, g_user_text
    ):
        """モーダル特徴の L2 正則化損失"""
        feat_reg = (
            1.0 / 2 * (g_item_image**2).sum()
            + 1.0 / 2 * (g_item_text**2).sum()
            + 1.0 / 2 * (g_user_image**2).sum()
            + 1.0 / 2 * (g_user_text**2).sum()
        )
        feat_reg = feat_reg / self.n_items
        feat_emb_loss = args.feat_reg_decay * feat_reg
        return feat_emb_loss

    def prune_loss(self, pred, drop_rate):
        """Hard example pruning: 低損失サンプルのみ残す (robust化)"""
        ind_sorted = np.argsort(pred.cpu().data).cuda()
        loss_sorted = pred[ind_sorted]
        remember_rate = 1 - drop_rate
        num_remember = int(remember_rate * len(loss_sorted))
        ind_update = ind_sorted[:num_remember]
        loss_update = pred[ind_update]
        return loss_update.mean()

    def mse_criterion(self, x, y, alpha=3):
        """正規化後 MSE 損失"""
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        tmp_loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
        tmp_loss = tmp_loss.mean()
        loss = F.mse_loss(x, y)
        return loss

    def sce_criterion(self, x, y, alpha=1):
        """Simple Cosine Error: (1 - cos)^alpha"""
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
        loss = loss.mean()
        return loss

    def test(self, users_to_test, is_val):
        """評価: 推論結果埋め込みを取得し test_torch で指標計算"""
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
        result = test_torch(ua_embeddings, ia_embeddings, users_to_test, is_val)
        return result

    def train(self):
        """学習ループ
        - サンプリング & LLM拡張サンプル追加
        - モーダル/属性BPR損失
        - 特徴正則化 + 属性再構成
        - 早期終了監視
        """

        now_time = datetime.now()
        run_time = datetime.strftime(now_time, "%Y_%m_%d__%H_%M_%S")

        training_time_list = []
        stopping_step = 0

        n_batch = data_generator.n_train // args.batch_size + 1
        best_recall = 0
        for epoch in range(args.epoch):
            t1 = time()
            loss, mf_loss, emb_loss, reg_loss = 0.0, 0.0, 0.0, 0.0
            contrastive_loss = 0.0
            n_batch = data_generator.n_train // args.batch_size + 1
            sample_time = 0.0
            build_item_graph = True
            # epoch 先頭で学習率スケジューリング等を行う場合はここに追記 (現状固定)
            # メモ: GPUメモリ利用は forward/backward でピーク; 逐次 del により圧迫緩和

            self.gene_u, self.gene_real, self.gene_fake = None, None, {}
            self.topk_p_dict, self.topk_id_dict = {}, {}

            for idx in tqdm(range(n_batch)):
                self.model_mm.train()
                sample_t1 = time()
                users, pos_items, neg_items = data_generator.sample()
                # data_generator.sample(): BPR 用 (u, pos_i, neg_i)
                #   - users: List[int] 長さ = 実バッチサイズ (args.batch_size)
                #   - pos_items: 正例 (観測インタラクション)
                #   - neg_items: 負例 (未観測アイテムからサンプリング)
                # 注意: 重複ユーザが存在しうる (サンプリング戦略依存)

                # --- 拡張サンプル (LLM生成疑似ペア) を一定割合で混入 ---
                augmented_sample_dict = pickle.load(
                    open(args.data_path + args.dataset + "/augmented_sample_dict", "rb")
                )
                users_aug = random.sample(users, int(len(users) * args.aug_sample_rate))
                # aug_sample_rate=0.2 の場合: 既存ユーザ20% を拡張例生成対象に
                pos_items_aug = [
                    augmented_sample_dict[user][0]
                    for user in users_aug
                    if (
                        augmented_sample_dict[user][0] < self.n_items
                        and augmented_sample_dict[user][1] < self.n_items
                    )
                ]
                neg_items_aug = [
                    augmented_sample_dict[user][1]
                    for user in users_aug
                    if (
                        augmented_sample_dict[user][0] < self.n_items
                        and augmented_sample_dict[user][1] < self.n_items
                    )
                ]
                users_aug = [
                    user
                    for user in users_aug
                    if (
                        augmented_sample_dict[user][0] < self.n_items
                        and augmented_sample_dict[user][1] < self.n_items
                    )
                ]
                self.new_batch_size = len(users_aug)
                users += users_aug
                pos_items += pos_items_aug
                neg_items += neg_items_aug
                # ここでバッチサイズが変動 (拡張分だけ増える) → 後続 loss は実際の件数で平均されない点に留意
                #  厳密な正規化が必要なら拡張前後で重み付けを導入可能

                sample_time += time() - sample_t1
                # --- 前向き伝播: 埋め込み/モーダル/属性/マスク対象出力 ---
                # model_mm 返り値 (想定形状):
                #   user_presentation_h: [n_users, d] 最終ユーザ表現 (融合後)
                #   item_presentation_h: [n_items, d] 最終アイテム表現
                #   image_i_feat, text_i_feat: [n_items, d_modal]
                #   image_u_feat, text_u_feat: [n_users, d_modal]
                #   user_prof_feat_pre / item_prof_feat_pre: 前段階(生)プロフィール表現
                #   user_prof_feat / item_prof_feat: 属性統合後プロフィール表現
                #   user_att_feats: ユーザ側属性辞書 (key -> tensor[n_users, d_attr])
                #   item_att_feats: アイテム側属性辞書 (key -> tensor[n_items, d_attr])
                #   i_mask_nodes / u_mask_nodes: マスク対象 index (再構成損失用)
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
                ) = self.model_mm(
                    self.ui_graph,
                    self.iu_graph,
                    self.image_ui_graph,
                    self.image_iu_graph,
                    self.text_ui_graph,
                    self.text_iu_graph,
                )

                # --- 基本BPR損失 ---
                u_bpr_emb = user_presentation_h[users]
                i_bpr_pos_emb = item_presentation_h[pos_items]
                i_bpr_neg_emb = item_presentation_h[neg_items]
                batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(
                    u_bpr_emb, i_bpr_pos_emb, i_bpr_neg_emb
                )
                # batch_mf_loss: BPR主損失 (hard pruning 後)
                # batch_emb_loss: L2 正則化 (self.decay 係数)
                # batch_reg_loss: 現状0 (拡張用プレースホルダ)

                # --- モーダル別 (画像/テキスト) BPR ---
                image_u_bpr_emb = image_u_feat[users]
                image_i_bpr_pos_emb = image_i_feat[pos_items]
                image_i_bpr_neg_emb = image_i_feat[neg_items]
                image_batch_mf_loss, image_batch_emb_loss, image_batch_reg_loss = (
                    self.bpr_loss(
                        image_u_bpr_emb, image_i_bpr_pos_emb, image_i_bpr_neg_emb
                    )
                )
                text_u_bpr_emb = text_u_feat[users]
                text_i_bpr_pos_emb = text_i_feat[pos_items]
                text_i_bpr_neg_emb = text_i_feat[neg_items]
                text_batch_mf_loss, text_batch_emb_loss, text_batch_reg_loss = (
                    self.bpr_loss(
                        text_u_bpr_emb, text_i_bpr_pos_emb, text_i_bpr_neg_emb
                    )
                )
                mm_mf_loss = image_batch_mf_loss + text_batch_mf_loss
                # mm_mf_loss: マルチモーダル (画像+テキスト) の BPR 主損失合計 (正則化は未使用)

                # --- 属性別BPR (アイテム属性辞書key毎) ---
                batch_mf_loss_aug = 0
                for index, value in enumerate(item_att_feats):  #
                    u_g_embeddings_aug = user_prof_feat[users]
                    pos_i_g_embeddings_aug = item_att_feats[value][pos_items]
                    neg_i_g_embeddings_aug = item_att_feats[value][neg_items]
                    tmp_batch_mf_loss_aug, batch_emb_loss_aug, batch_reg_loss_aug = (
                        self.bpr_loss(
                            u_g_embeddings_aug,
                            pos_i_g_embeddings_aug,
                            neg_i_g_embeddings_aug,
                        )
                    )
                    batch_mf_loss_aug += tmp_batch_mf_loss_aug
                # batch_mf_loss_aug: 属性軸のランキング信号 (複数属性を合算) → 後で aug_mf_rate を掛ける

                # --- 特徴量正則化 ---
                feat_emb_loss = self.feat_reg_loss_calculation(
                    image_i_feat, text_i_feat, image_u_feat, text_u_feat
                )
                # feat_emb_loss: モーダル埋め込み全体のノルム抑制 (過学習/発散防止)

                # --- 属性再構成損失 (マスク有効時) ---
                att_re_loss = 0
                if args.mask:
                    input_i = {}
                    for index, value in enumerate(item_att_feats.keys()):
                        input_i[value] = item_att_feats[value][i_mask_nodes]
                    decoded_u, decoded_i = self.decoder(
                        torch.tensor(user_prof_feat[u_mask_nodes]), input_i
                    )
                    if args.feat_loss_type == "mse":
                        att_re_loss += self.mse_criterion(
                            decoded_u,
                            torch.tensor(self.user_init_embedding[u_mask_nodes]).cuda(),
                            alpha=args.alpha_l,
                        )
                        # decoded_i: list[Tensor] 順序 = item_att_feats.keys() の列挙順
                        for index, value in enumerate(item_att_feats.keys()):
                            att_re_loss += self.mse_criterion(
                                decoded_i[index],
                                torch.tensor(
                                    self.item_attribute_embedding[value][i_mask_nodes]
                                ).cuda(),
                                alpha=args.alpha_l,
                            )
                    elif args.feat_loss_type == "sce":
                        att_re_loss += self.sce_criterion(
                            decoded_u,
                            torch.tensor(self.user_init_embedding[u_mask_nodes]).cuda(),
                            alpha=args.alpha_l,
                        )
                        for index, value in enumerate(item_att_feats.keys()):
                            att_re_loss += self.sce_criterion(
                                decoded_i[index],
                                torch.tensor(
                                    self.item_attribute_embedding[value][i_mask_nodes]
                                ).cuda(),
                                alpha=args.alpha_l,
                            )
                # att_re_loss: マスク再構成 (ユーザ初期埋め込み + 各属性埋め込み) の類似性損失
                # alpha_l: cos誤差指数 or MSE正規化の強度パラメータ

                # --- 総合損失 ---
                batch_loss = (
                    batch_mf_loss
                    + batch_emb_loss
                    + batch_reg_loss
                    + feat_emb_loss
                    + args.aug_mf_rate * batch_mf_loss_aug
                    + args.mm_mf_rate * mm_mf_loss
                    + args.att_re_rate * att_re_loss
                )
                nn.utils.clip_grad_norm_(
                    self.model_mm.parameters(), max_norm=1.0
                )  # 勾配爆発抑制
                self.optimizer.zero_grad()
                batch_loss.backward(retain_graph=False)
                # retain_graph=False: グラフを都度解放しメモリ節約 (再利用不要)

                self.optimizer.step()
                # optimizer.step(): AdamW により weight decay を decoupled に適用

                loss += float(batch_loss)
                mf_loss += float(batch_mf_loss)
                emb_loss += float(batch_emb_loss)
                reg_loss += float(batch_reg_loss)
                # ここでログを逐次出したい場合は条件 (idx % log_interval == 0) を導入

            del (
                user_presentation_h,
                item_presentation_h,
                u_bpr_emb,
                i_bpr_neg_emb,
                i_bpr_pos_emb,
            )
            # del: 大規模 [n_users/n_items, d] テンソル参照を解放し次epoch前にGC促進

            if math.isnan(loss) == True:
                self.logger.logging("ERROR: loss is nan.")
                sys.exit()

            if (epoch + 1) % args.verbose != 0:
                perf_str = (
                    "Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f  + %.5f]"
                    % (
                        epoch,
                        time() - t1,
                        loss,
                        mf_loss,
                        emb_loss,
                        reg_loss,
                        contrastive_loss,
                    )
                )
                training_time_list.append(time() - t1)
                self.logger.logging(perf_str)

            t2 = time()
            users_to_test = list(data_generator.test_set.keys())
            users_to_val = list(data_generator.val_set.keys())
            ret = self.test(users_to_test, is_val=False)  # ^-^
            training_time_list.append(t2 - t1)

            t3 = time()

            if args.verbose > 0:
                perf_str = (
                    "Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f], recall=[%.5f, %.5f, %.5f, %.5f], "
                    "precision=[%.5f, %.5f, %.5f, %.5f], hit=[%.5f, %.5f, %.5f, %.5f], ndcg=[%.5f, %.5f, %.5f, %.5f]"
                    % (
                        epoch,
                        t2 - t1,
                        t3 - t2,
                        loss,
                        mf_loss,
                        emb_loss,
                        reg_loss,
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
                self.logger.logging(perf_str)

            # 早期終了判定: 指定K(例: K=?) のRecall改善で更新
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
                # best_recall は Ks の2番目 (例: K=10) の改善のみ追跡 (必要なら全Kに拡張可)
                stopping_step = 0
            elif stopping_step < args.early_stopping_patience:
                stopping_step += 1
                self.logger.logging(
                    "#####Early stopping steps: %d #####" % stopping_step
                )
            else:
                self.logger.logging("#####Early stop! #####")
                break
        self.logger.logging(str(test_ret))

        return best_recall, run_time

    def bpr_loss(self, users, pos_items, neg_items):
        """BPR損失
        prune_loss により一部サンプルをdropしロバスト化
        returns: mf_loss, emb_loss, reg_loss
        """
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
        # pos_scores - neg_scores が 大きいほど "正例 > 負例" を満たす (ランキング改善)

        regularizer = (
            1.0 / (2 * (users**2).sum() + 1e-8)
            + 1.0 / (2 * (pos_items**2).sum() + 1e-8)
            + 1.0 / (2 * (neg_items**2).sum() + 1e-8)
        )
        regularizer = regularizer / self.batch_size
        # 通常の L2( ||U||^2 + ||I+||^2 + ||I-||^2 ) ではなく逆数和 → ノルムが大きいほど寄与減少 (挙動: scale 抑制)

        maxi = F.logsigmoid(pos_scores - neg_scores + 1e-8)
        mf_loss = -self.prune_loss(maxi, args.prune_loss_drop_rate)
        # prune_loss: logσ を昇順ソート (小さい=難例) を優先学習 (drop_rate 分の易しい例を排除)

        emb_loss = self.decay * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """scipy疎行列 -> torch.sparse.FloatTensor (CPU)"""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
        )
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)


def set_seed(seed):
    """再現性確保のためのシード固定"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    # エントリポイント: 環境設定 & Trainer実行
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    set_seed(args.seed)
    config = dict()
    config["n_users"] = data_generator.n_users
    config["n_items"] = data_generator.n_items

    trainer = Trainer(data_config=config)
    trainer.train()
