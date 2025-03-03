import numpy as np
import cv2
import json

with open("keypoints_2d.json", "r") as f:
    keypoints_2d = json.load(f)

params = np.load("camera_calibration_params.npz", allow_pickle=True)
camera_matrices = params["camera_matrices"]
rvecs_list = params["rvecs_list"]
tvecs_list = params["tvecs_list"]

P_matrices = []
num_cameras = len(camera_matrices)

for i in range(num_cameras):
    rvec = np.mean(np.array(rvecs_list[i]), axis=0)
    tvec = np.mean(np.array(tvecs_list[i]), axis=0)

    R, _ = cv2.Rodrigues(rvec)
    P = np.dot(camera_matrices[i], np.hstack((R, tvec)))
    P_matrices.append(P)


keypoints_3d = []
confidence_threshold = 0.0
print(len(keypoints_2d["cam1"]))
print(len(keypoints_2d["cam1"][0]))
for frame_idx in range(len(keypoints_2d["cam1"])):
    points_3d = []

    for kp_idx in range(len(keypoints_2d["cam1"][frame_idx])):
        try:
            p1_data = keypoints_2d["cam1"][frame_idx][kp_idx]
            p2_data = keypoints_2d["cam2"][frame_idx][kp_idx]
            p3_data = keypoints_2d["cam3"][frame_idx][kp_idx]
            p4_data = keypoints_2d["cam4"][frame_idx][kp_idx]

            p1 = np.array(p1_data[:2], dtype=np.float32)
            p2 = np.array(p2_data[:2], dtype=np.float32)
            p3 = np.array(p3_data[:2], dtype=np.float32)
            p4 = np.array(p4_data[:2], dtype=np.float32)

            conf1 = p1_data[2]
            conf2 = p2_data[2]
            conf3 = p3_data[2]
            conf4 = p4_data[2]

            if conf1 < confidence_threshold or conf2 < confidence_threshold or conf3 < confidence_threshold or conf4 < confidence_threshold:
                print(f"⚠️ frame {frame_idx} : keypoint {kp_idx} is skipped due to low accuracy")
                continue

            # triangulation
            points_4d = cv2.triangulatePoints(P_matrices[0], P_matrices[1], p1.reshape(2, 1), p2.reshape(2, 1))
            points_3d.append((points_4d[:3] / points_4d[3]).flatten().tolist())

        except Exception as e:
            print(f"⚠️ frame {frame_idx} : keypoint {kp_idx} error: {e}")
            points_3d.append([])

    keypoints_3d.append(points_3d)

with open("keypoints_3d.json", "w") as f:
    json.dump(keypoints_3d, f, indent=4)


print("✅ 3D keypoint.json saved！")
