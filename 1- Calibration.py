import cv2
import numpy as np
import os

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()

marker_length = 0.4

video_paths = [
    "calib_mov/Record_QRCalibration_new_WebcamFullLeft.avi",
    "calib_mov/Record_QRCalibration_new_WebcamFullRight.avi",
    "calib_mov/Record_QRCalibration_new_WebcamLeft.avi",
    "calib_mov/Record_QRCalibration_new_WebcamRight.avi"
]

camera_matrices = []
distortion_coeffs = []
rvecs_list = []
tvecs_list = []

for video_path in video_paths:
    cap = cv2.VideoCapture(video_path)

    objpoints = []
    imgpoints = []
    frame_shape = None
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 10 != 0:
            continue

        frame_shape = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame_count % 100 == 0:
            print(f"🟢 {video_path}: {frame_count} frame processing...")

        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

        if ids is not None and len(corners) > 0:
            for i in range(len(ids)):
                imgpoints.append(corners[i].reshape(-1, 2))

                objp = np.array([
                    [-marker_length / 2,  marker_length / 2, 0],
                    [ marker_length / 2,  marker_length / 2, 0],
                    [ marker_length / 2, -marker_length / 2, 0],
                    [-marker_length / 2, -marker_length / 2, 0]
                ], dtype=np.float32)

                objpoints.append(objp)

    cap.release()

    if frame_shape is None:
        print(f"❌ The frame in {video_path} could not be loaded correctly.")
        continue

    if len(objpoints) == 0 or len(imgpoints) == 0:
        print(f"❌ {video_path}: ArUco marker not detected and cannot be calibrated")
        continue

    if len(objpoints) > 100:
        objpoints = objpoints[:100]
        imgpoints = imgpoints[:100]

    assert len(objpoints) == len(imgpoints), f"❌ {video_path}: objpoints and imgpoints sizes do not match"

    print(f"⏳ {video_path}: Calibration in progress...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frame_shape[::-1], None, None)

    camera_matrices.append(mtx)
    distortion_coeffs.append(dist)
    rvecs_list.append(rvecs)
    tvecs_list.append(tvecs)

    print(f"✅ Success Calibration!: {video_path}")

np.savez("camera_calibration_params.npz",
         camera_matrices=camera_matrices,
         distortion_coeffs=distortion_coeffs,
         rvecs_list=rvecs_list,
         tvecs_list=tvecs_list)

print("✅ Calibration of all cameras completed!！")
