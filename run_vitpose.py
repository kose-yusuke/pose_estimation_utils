import cv2
import torch
import numpy as np
import os
import json
from mmpose.apis import init_pose_model, inference_top_down_pose_model

# 設定ファイルとチェックポイントを指定
config_file = 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_simple_coco_256x192.py'
checkpoint_file = 'pretrained_models/vitpose-h-simple.pth'

# 使用するデバイスを設定
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pose_model = init_pose_model(config_file, checkpoint_file, device=device)

# 動画のパス
video_path = 'demo/resources/1088_WebcamFullRight_trial0_IDsFiltered_clean.mp4'
cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# JSON のパス（動画名に対応する JSON を取得）
json_path = '1088_WebcamFullRight_trial0_IDsFiltered_clean.json'

# JSON をロード
with open(json_path, "r") as f:
    bbox_data = json.load(f)

# キーポイントの保存リスト
keypoints_2d = []
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # バウンディングボックス情報を取得
    frame_idx_str = str(frame_idx)  # フレーム番号を文字列化
    if frame_idx_str in bbox_data:
        bbox_list = bbox_data[frame_idx_str]  # フレームごとのバウンディングボックス情報を取得
        bboxes = [entry["body_xy"] for entry in bbox_list]  # bbox座標をリストに変換
    else:
        bboxes = []

    # 修正: bboxes を ViTPose の期待するフォーマットに変換
    bboxes = [{"bbox": bbox} for bbox in bboxes]

    # # デバッグ出力
    # print(f"Frame {frame_idx} - Bboxes: {bboxes}")

    # OpenCVのBGR画像をRGBに変換
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # mmpose で推論
    results = inference_top_down_pose_model(pose_model, frame_rgb, bboxes)
    # print(f"Frame {frame_idx} - Results: {results}")  # デバッグ出力

    # 取得したキーポイントを保存
    frame_keypoints = []
    if results and isinstance(results, tuple):  # tupleの場合は最初の要素を取得
        results = results[0]

    if isinstance(results, list):  # 結果がリストなら処理
        for res in results:
            if isinstance(res, dict) and "keypoints" in res:
                keypoints = res["keypoints"].tolist()
                frame_keypoints.append(keypoints)

    keypoints_2d.append({"frame": frame_idx, "keypoints": frame_keypoints})
    frame_idx += 1

cap.release()

# JSON ファイルに保存
output_json = "keypoints_2d.json"
with open(output_json, 'w') as f:
    json.dump(keypoints_2d, f)

print(f"✅ 2D Keypoint extraction complete! Saved to {output_json}")
