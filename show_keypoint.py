import cv2
import json
import torch
import numpy as np
from mmpose.apis import init_pose_model, inference_top_down_pose_model

# ✅ **より精度の高い WholeBody モデルを使用**
# config_file = 'configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/ViTPose_huge_wholebody_256x192.py'
# checkpoint_file = 'pretrained_models/vitpose_huge_wholebody.pth'

config_file = 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_simple_coco_256x192.py'
checkpoint_file = 'pretrained_models/vitpose-h-simple.pth'

# モデルのロード
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pose_model = init_pose_model(config_file, checkpoint_file, device=device)

# 動画と JSON のパス
video_path = "demo/resources/3409_WebcamRight_trial0_IDsFiltered_clean.mp4"
json_path = "test_json/3409_WebcamRight_trial0_IDsFiltered_clean.json"

# JSON のロード
with open(json_path, "r") as f:
    bbox_data = json.load(f)

cap = cv2.VideoCapture(video_path)
frame_idx = 0

# **動画の情報を取得**
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# ✅ **動画の保存設定**
output_path = "test_mov/Right_02_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4v または XVID
scale = 0.5  # 縮小スケール
out_width, out_height = int(frame_width * scale), int(frame_height * scale)
out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

# **キーポイントの接続関係（WholeBody用に拡張）**
wholebody_pairs = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # 右腕
    (0, 5), (5, 6), (6, 7), (7, 8),  # 左腕
    (0, 9), (9, 10), (10, 11),  # 体幹 (首〜骨盤)
    (11, 12), (12, 13), (13, 14),  # 右脚
    (11, 15), (15, 16), (16, 17),  # 左脚
    (0, 18), (18, 19), (19, 20), (20, 21),  # 顔
    (0, 22), (22, 23), (23, 24), (24, 25)  # 手
]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # **バウンディングボックス情報を取得**
    frame_idx_str = str(frame_idx)  # フレーム番号を文字列化
    if frame_idx_str in bbox_data:
        bbox_list = bbox_data[frame_idx_str]
        bboxes = [{"bbox": entry["body_xy"]} for entry in bbox_list]
    else:
        bboxes = []

    # **バウンディングボックス外を黒塗り**
    mask = np.zeros_like(frame, dtype=np.uint8)
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox["bbox"]
        mask[y1:y2, x1:x2] = frame[y1:y2, x1:x2]  # バウンディングボックス内は元の画像を保持

    # OpenCVのBGR画像をRGBに変換
    frame_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    # **ViTPose で推論**
    results = inference_top_down_pose_model(pose_model, frame_rgb, bboxes)

    # **バウンディングボックスの描画**
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 緑色のボックス

    # **キーポイントの描画**
    if isinstance(results, tuple):
        results = results[0]  # `inference_top_down_pose_model` がタプルを返す場合の処理

    if isinstance(results, list):  # `results` がリストであることを確認
        for res in results:
            if isinstance(res, dict) and "keypoints" in res:
                keypoints = res["keypoints"]

                # **キーポイントを線で結ぶ**
                for pair in wholebody_pairs:
                    partA, partB = pair
                    if partA < len(keypoints) and partB < len(keypoints):
                        xA, yA, confA = keypoints[partA]
                        xB, yB, confB = keypoints[partB]
                        if confA > 0.5 and confB > 0.5:  # 両方のキーポイントの信頼度が0.5以上なら線を描画
                            cv2.line(frame, (int(xA), int(yA)), (int(xB), int(yB)), (255, 0, 0), 2)

                # **キーポイントを点で描画**
                for kp in keypoints:
                    x, y, conf = kp
                    if conf > 0.5:  # 信頼度が 0.5 以上のものだけ描画
                        cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)  # 赤色の点
                    else:
                        cv2.circle(frame, (int(x), int(y)), 5, (255, 0, 0), -1)  # 青色の点（低信頼度）

    # **動画のサイズを縮小**
    frame_resized = cv2.resize(frame, (out_width, out_height))

    # **動画を保存**
    out.write(frame_resized)

    # **画面に表示**
    cv2.imshow("Pose Estimation", frame_resized)
    if cv2.waitKey(30) & 0xFF == 27:
        break

    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ 動画保存完了: {output_path}")
