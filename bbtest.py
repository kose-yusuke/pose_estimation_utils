import numpy as np

# npyファイルのロード
bbox_data = np.load("0215_WebcamFullLeft_trial0_IDsFiltered_clean.npy", allow_pickle=True)
#
# # データの中身を確認
# print(bbox_data.shape)  # 例: (300, 5) → 300フレーム分のデータ
# print(bbox_data[:5])    # 最初の5つのデータを表示
