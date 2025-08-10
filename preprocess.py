# 必要パッケージ
# pip install pandas numpy scipy

import pickle
import re

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from torch_geometric.datasets import MovieLens

# === 0) データセットのダウンロード ===
ml = MovieLens(root="data/movielens")

# === 1) 読み込み（MovieLens raw） ===
movies = pd.read_csv(
    "data/movielens/raw/ml-latest-small/movies.csv"
)  # movieId,title,genres
ratings = pd.read_csv(
    "data/movielens/raw/ml-latest-small/ratings.csv"
)  # userId,movieId,rating,timestamp
# 公式READMEのとおり 1〜5 の星レーティングです。:contentReference[oaicite:3]{index=3}

# === 2) 連番IDマッピング（内部用ID） ===
#   ・movieId と userId を 0..M-1 / 0..U-1 の連番に圧縮
#   ・以後、あなたの全ファイルはこの連番で統一します
unique_movie = np.sort(ratings["movieId"].unique())
unique_user = np.sort(ratings["userId"].unique())
movieId_to_idx = {mid: i for i, mid in enumerate(unique_movie)}
userId_to_idx = {uid: i for i, uid in enumerate(unique_user)}
M = len(unique_movie)
U = len(unique_user)

# ratings に連番列を追加
ratings["u"] = ratings["userId"].map(userId_to_idx)
ratings["i"] = ratings["movieId"].map(movieId_to_idx)

# === 3) item_attribute.csv を作る ===
# (A) id,title,genre 版
movies_in_use = movies[movies["movieId"].isin(unique_movie)].copy()
movies_in_use["id"] = movies_in_use["movieId"].map(movieId_to_idx)
item_attr_title_genre = (
    movies_in_use[["id", "title", "genres"]].sort_values("id").set_index("id")
)
# ヘッダ無し・index込みで書き出し（あなたの loader は header=None で読む想定）
item_attr_title_genre.to_csv("item_attribute.csv", header=False)


# (B) id,year,title 版（必要なら）
def parse_year(title: str):
    m = re.search(r"\((\d{4})\)\s*$", str(title))
    return int(m.group(1)) if m else np.nan


movies_in_use["year"] = movies_in_use["title"].map(parse_year)
item_attr_year_title = (
    movies_in_use[["id", "year", "title"]].sort_values("id").set_index("id")
)
# 上書きしたくないなら別名で出力
# item_attr_year_title.to_csv("item_attribute_year_title.csv", header=False)

# === 4) train_mat を作る（視聴/接触の有無で 1 を立てる） ===
# 明示評価(1〜5)の有無でバイナリ化。履歴としては全部1でOK。
rows = ratings["u"].to_numpy()
cols = ratings["i"].to_numpy()
data = np.ones_like(rows, dtype=np.float32)
train_mat = csr_matrix((data, (rows, cols)), shape=(U, M))
with open("train_mat", "wb") as f:
    pickle.dump(train_mat, f)

# === 5) candidate_indices を作る ===
K = 100  # 例：各ユーザに100件候補
# 人気度 = 評価件数で降順
pop_counts = ratings.groupby("i").size().sort_values(ascending=False)
popular_items = pop_counts.index.to_numpy()  # 連番IDの配列（人気順）

# 各ユーザの既視聴（連番ID）の set
user_hist = ratings.groupby("u")["i"].apply(lambda s: set(s.to_list()))

candidates = np.zeros((U, K), dtype=np.int64)
for u in range(U):
    # 未視聴アイテムを人気順でフィルタして先頭K件
    unseen = [i for i in popular_items if i not in user_hist[u]]
    # 人気上位だけでKが足りなければ、残りはランダムサンプル等で埋めてもOK
    take = (
        unseen[:K]
        if len(unseen) >= K
        else unseen + list(np.random.choice(M, K - len(unseen), replace=False))
    )
    candidates[u, :] = np.array(take[:K], dtype=np.int64)

with open("candidate_indices", "wb") as f:
    pickle.dump(candidates, f)

# === 6) augmented_sample_dict を初期化 ===
with open("augmented_sample_dict", "wb") as f:
    pickle.dump({}, f)
