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

# ✅ **動画と JSON のパス**
video_path = "demo/resources/3409_WebcamRight_trial0_IDsFiltered_clean.mp4"
json_path = "test_json/3409_WebcamRight_trial0_IDsFiltered_clean.json"

# ✅ **JSON のロード**
with open(json_path, "r") as f:
    bbox_data = json.load(f)

cap = cv2.VideoCapture(video_path)
frame_idx = 0

# ✅ **動画の情報を取得**
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# ✅ **動画の保存設定**
output_path = "test_mov/Right_02_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
scale = 0.5
out_width, out_height = int(frame_width * scale), int(frame_height * scale)
out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

# ✅ **キーポイントの接続関係**
wholebody_pairs = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # 右腕
    (0, 5), (5, 6), (6, 7), (7, 8),  # 左腕
    (0, 9), (9, 10), (10, 11),  # 体幹
    (11, 12), (12, 13), (13, 14),  # 右脚
    (11, 15), (15, 16), (16, 17),  # 左脚
]

# ✅ **セグメンテーション用の変換関数**
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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

    # ✅ **バウンディングボックスの描画**
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox["bbox"]
        cv2.rectangle(black_background, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # ✅ **キーポイントの描画**
    if isinstance(results, tuple):
        results = results[0]

    if isinstance(results, list):
        for res in results:
            if isinstance(res, dict) and "keypoints" in res:
                keypoints = res["keypoints"]

                for pair in wholebody_pairs:
                    partA, partB = pair
                    if partA < len(keypoints) and partB < len(keypoints):
                        xA, yA, confA = keypoints[partA]
                        xB, yB, confB = keypoints[partB]
                        if confA > 0.5 and confB > 0.5:
                            cv2.line(black_background, (int(xA), int(yA)), (int(xB), int(yB)), (255, 0, 0), 2)

                for kp in keypoints:
                    x, y, conf = kp
                    if conf > 0.5:
                        cv2.circle(black_background, (int(x), int(y)), 5, (0, 0, 255), -1)

    # ✅ **動画のサイズを縮小**
    frame_resized = cv2.resize(black_background, (out_width, out_height))

    # ✅ **動画を保存**
    out.write(frame_resized)

    # ✅ **画面に表示**
    cv2.imshow("Pose Estimation", frame_resized)
    if cv2.waitKey(30) & 0xFF == 27:
        break

    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ 動画保存完了: {output_path}")
