# import json
#
# # JSONを読み込む
# with open("keypoints_2d.json", "r") as f:
#     keypoints_2d = json.load(f)
#
# # 🔍 構造確認
# # print(keypoints_2d.shape())
# print("🔍 keypoints_2d['cam1'] の最初のフレームデータ:")
# print(json.dumps(keypoints_2d["cam1"][0], indent=4))

import json

# 3Dキーポイントをロード
with open("keypoints_3d.json", "r") as f:
    keypoints_3d = json.load(f)

# 🔍 最初の3フレームのデータ構造を確認
for i in range(min(3, len(keypoints_3d))):
    print(f"🔍 フレーム {i} のデータ構造:")
    print(json.dumps(keypoints_3d[i], indent=4))
    print("\n" + "="*50 + "\n")  # 区切り

# for i in range(min(3, len(keypoints_3d))):
#     print(f"🔍 フレーム {i} の型: {type(keypoints_3d[i])}")
#     if isinstance(keypoints_3d[i], list):
#         if len(keypoints_3d[i]) > 0:
#             print(f"    🔹 フレーム {i} 内の要素の型: {type(keypoints_3d[i][0])}")
#             if isinstance(keypoints_3d[i][0], list) and len(keypoints_3d[i][0]) > 0:
#                 print(f"    🔸 最初のキーポイントの型: {type(keypoints_3d[i][0][0])}")
#     print("\n" + "="*50 + "\n")
