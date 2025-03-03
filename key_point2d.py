import cv2
import json
import torch
import numpy as np
from mmpose.apis import init_pose_model, inference_top_down_pose_model
from torchvision import models
import torchvision.transforms as T

segmentation_model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

config_file = 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_simple_coco_256x192.py'
checkpoint_file = 'pretrained_models/vitpose-h-simple.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pose_model = init_pose_model(config_file, checkpoint_file, device=device)

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

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

keypoints_2d = {cam: [] for cam in video_paths.keys()}

for cam_id, (cam_name, video_path) in enumerate(video_paths.items()):
    print(f"🔵 {cam_name} processing started...")

    cap = cv2.VideoCapture(video_path)

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

        black_background = np.zeros_like(frame, dtype=np.uint8)

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox["bbox"]

            bbox_frame = frame[y1:y2, x1:x2]

            bbox_rgb = cv2.cvtColor(bbox_frame, cv2.COLOR_BGR2RGB)
            input_tensor = transform(bbox_rgb).unsqueeze(0)

            with torch.no_grad():
                output = segmentation_model(input_tensor)["out"][0]

            seg_map = output.argmax(0).byte().cpu().numpy()
            person_mask = (seg_map == 15).astype(np.uint8) * 255

            bbox_segmented = cv2.bitwise_and(bbox_frame, bbox_frame, mask=person_mask)

            black_background[y1:y2, x1:x2] = bbox_segmented

        results = inference_top_down_pose_model(pose_model, black_background, bboxes)

        if isinstance(results, tuple):
            results = results[0]

        frame_keypoints = []
        if isinstance(results, list):
            for res in results:
                if isinstance(res, dict) and "keypoints" in res:
                    keypoints = res["keypoints"]
                    frame_keypoints.append(keypoints.tolist())

        if frame_keypoints:
            keypoints_2d[cam_name].append(frame_keypoints[0])  # 余計なネストを削除
        else:
            keypoints_2d[cam_name].append([])

        frame_idx += 1

    cap.release()
    print(f"✅ {cam_name} processing completed！（The amount of frames: {frame_idx}）")

with open("keypoints_2d.json", "w") as f:
    json.dump(keypoints_2d, f, indent=4)

print("✅ Completed saving 2D key-points of 4 viewpoints！")
