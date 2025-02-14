import cv2
import json
import numpy as np
import matplotlib.pyplot as plt

# JSONデータの読み込み
keypoints_json = "keypoints_2d.json"
video_path = "demo/resources/1088_WebcamFullRight_trial0_IDsFiltered_clean.mp4"
output_video_path = "output.mp4"

with open(keypoints_json, "r") as f:
    keypoints_data = json.load(f)

# 動画の読み込み
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 出力動画の設定
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# キーポイントの接続情報 (COCOフォーマットに基づく)
skeleton_connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # 頭〜腕
    (0, 5), (5, 6), (6, 7), (7, 8),  # 胴体
    (5, 9), (9, 10), (10, 11),       # 右腕
    (6, 12), (12, 13), (13, 14),     # 左腕
    (11, 15), (15, 16),              # 右脚
    (14, 17), (17, 18)               # 左脚
]

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # キーポイント情報を取得
    if frame_idx < len(keypoints_data):
        frame_info = keypoints_data[frame_idx]
        keypoints = frame_info["keypoints"]

        if keypoints:
            for person_keypoints in keypoints:
                keypoints_np = np.array(person_keypoints)  # NumPy配列に変換
                for (i, j) in skeleton_connections:
                    if i < len(keypoints_np) and j < len(keypoints_np):
                        pt1 = tuple(map(int, keypoints_np[i][:2]))
                        pt2 = tuple(map(int, keypoints_np[j][:2]))
                        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

                # 各キーポイントを描画
                for keypoint in keypoints_np:
                    x, y, conf = keypoint
                    if conf > 0.2:  # 信頼度が低すぎる点は描画しない
                        cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ キーポイント可視化完了！保存先: {output_video_path}")
