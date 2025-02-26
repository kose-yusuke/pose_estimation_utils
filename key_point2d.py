import cv2
import json
import torch
import numpy as np
from mmpose.apis import init_pose_model, inference_top_down_pose_model
from torchvision import models
import torchvision.transforms as T

# ✅ **セグメンテーション用の DeepLabV3 モデルをロード**
segmentation_model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

# ✅ **ViTPose モデルのロード**
config_file = 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_simple_coco_256x192.py'
checkpoint_file = 'pretrained_models/vitpose-h-simple.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pose_model = init_pose_model(config_file, checkpoint_file, device=device)

# ✅ **各カメラの動画とバウンディングボックスJSONのパス**
video_paths = {
    "cam1": "demo/resources/3409_WebcamFullLeft_trial0_IDsFiltered_clean.mp4",
    "cam2": "demo/resources/3409_WebcamFullRight_trial0_IDsFiltered_clean.mp4",
    "cam3": "demo/resources/3409_WebcamLeft_trial0_IDsFiltered_clean.mp4",
    "cam4": "demo/resources/3409_WebcamRight_trial0_IDsFiltered_clean.mp4"
}

json_paths = {
    "cam1": "test_json/3409_WebcamFullLeft_trial0_IDsFiltered_clean.json",
    "cam2": "test_json/3409_WebcamFullRight_trial0_IDsFiltered_clean.json",
    "cam3": "test_json/3409_WebcamLeft_trial0_IDsFiltered_clean.json",
    "cam4": "test_json/3409_WebcamRight_trial0_IDsFiltered_clean.json"
}

# ✅ **セグメンテーション用の変換関数**
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ✅ **キーポイントの保存用辞書**
keypoints_2d = {cam: [] for cam in video_paths.keys()}

# ✅ **各カメラの動画を処理**
for cam_id, (cam_name, video_path) in enumerate(video_paths.items()):
    print(f"🔵 {cam_name} の処理を開始...")

    cap = cv2.VideoCapture(video_path)

    # JSON のロード（バウンディングボックス）
    json_path = json_paths[cam_name]
    with open(json_path, "r") as f:
        bbox_data = json.load(f)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx_str = str(frame_idx)
        if frame_idx_str in bbox_data:
            bbox_list = bbox_data[frame_idx_str]
            bboxes = [{"bbox": entry["body_xy"]} for entry in bbox_list]
        else:
            bboxes = []

        # ✅ **黒背景の画像を作成**
        black_background = np.zeros_like(frame, dtype=np.uint8)

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox["bbox"]

            # **バウンディングボックスの領域を切り出し**
            bbox_frame = frame[y1:y2, x1:x2]

            # **セグメンテーションを適用**
            bbox_rgb = cv2.cvtColor(bbox_frame, cv2.COLOR_BGR2RGB)
            input_tensor = transform(bbox_rgb).unsqueeze(0)

            with torch.no_grad():
                output = segmentation_model(input_tensor)["out"][0]

            # **セグメンテーション結果から「人物（クラスID 15）」のみ抽出**
            seg_map = output.argmax(0).byte().cpu().numpy()
            person_mask = (seg_map == 15).astype(np.uint8) * 255  # 人物部分は白、それ以外は黒

            # **マスクを元の画像に適用（人物部分のみ残す）**
            bbox_segmented = cv2.bitwise_and(bbox_frame, bbox_frame, mask=person_mask)

            # **元のフレームに適用（黒背景の上にペースト）**
            black_background[y1:y2, x1:x2] = bbox_segmented

        # ✅ **ViTPose で推論（セグメンテーション済み画像を使用）**
        results = inference_top_down_pose_model(pose_model, black_background, bboxes)

        # ✅ **キーポイントの保存**
        if isinstance(results, tuple):
            results = results[0]

        frame_keypoints = []
        if isinstance(results, list):
            for res in results:
                if isinstance(res, dict) and "keypoints" in res:
                    keypoints = res["keypoints"]
                    frame_keypoints.append(keypoints.tolist())

        # ✅ 修正：リストのネストを削除
        if frame_keypoints:
            keypoints_2d[cam_name].append(frame_keypoints[0])  # 余計なネストを削除
        else:
            keypoints_2d[cam_name].append([])

        frame_idx += 1

    cap.release()
    print(f"✅ {cam_name} の処理が完了！（総フレーム数: {frame_idx}）")

# ✅ **2DキーポイントをJSONに保存**
with open("keypoints_2d.json", "w") as f:
    json.dump(keypoints_2d, f, indent=4)

print("✅ 4視点の2Dキーポイントを保存完了！")
