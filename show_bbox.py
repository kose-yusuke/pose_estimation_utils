import cv2
import json

video_path = "demo/resources/1088_WebcamFullRight_trial0_IDsFiltered_clean.mp4"
json_path = "1088_WebcamFullRight_trial0_IDsFiltered_clean.json"

# JSON のロード
with open(json_path, "r") as f:
    bbox_data = json.load(f)

cap = cv2.VideoCapture(video_path)

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if str(frame_idx) in bbox_data:
        for bbox in bbox_data[str(frame_idx)]:
            if "bbox" in bbox:
                x1, y1, x2, y2 = bbox["bbox"]
            elif "body_xy" in bbox:  # こちらのキーの可能性がある
                x1, y1, x2, y2 = bbox["body_xy"]
            else:
                print(f"⚠ Frame {frame_idx}: No bbox found in {bbox}")
                continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Bounding Boxes", frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
