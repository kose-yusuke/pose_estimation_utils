import cv2
import numpy as np
import os

# ArUco マーカーの辞書を定義
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()

# キャリブレーション用動画のパス
video_paths = [
    "calib_mov/Record_QRCalibration_new_WebcamFullLeft.avi",
    "calib_mov/Record_QRCalibration_new_WebcamFullRight.avi",
    "calib_mov/Record_QRCalibration_new_WebcamLeft.avi",
    "calib_mov/Record_QRCalibration_new_WebcamRight.avi"
]

# ArUcoマーカーのサイズ（メートル単位）
marker_length = 0.3  # 30cm

# 各カメラのパラメータを保存
camera_matrices = []
distortion_coeffs = []
rvecs_list = []
tvecs_list = []

for video_path in video_paths:
    cap = cv2.VideoCapture(video_path)

    objpoints = []  # 3D点リスト
    imgpoints = []  # 2D点リスト
    frame_shape = None  # 画像サイズを保存
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 10 != 0:  # 10フレームごとに処理
            continue

        frame_shape = frame.shape[:2]  # フレームサイズを保存
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ArUcoマーカーを検出（ROIなし）
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

        # デバッグ用: 50フレームごとに画像を保存
        if frame_count % 50 == 0:
            cv2.imwrite(f"debug_frame_{frame_count}.jpg", gray)

        if ids is not None and len(corners) > 0:
            for i in range(len(ids)):
                imgpoints.append(corners[i].reshape(-1, 2))

                # ArUcoマーカーの3D座標（ワールド座標系）
                objp = np.array([
                    [-marker_length / 2,  marker_length / 2, 0],  # 左上
                    [ marker_length / 2,  marker_length / 2, 0],  # 右上
                    [ marker_length / 2, -marker_length / 2, 0],  # 右下
                    [-marker_length / 2, -marker_length / 2, 0]   # 左下
                ], dtype=np.float32)

                objpoints.append(objp)

    cap.release()

    if frame_shape is None:
        print(f"❌ {video_path} のフレームが正しく読み込めませんでした")
        continue

    if len(objpoints) == 0 or len(imgpoints) == 0:
        print(f"❌ {video_path}: ArUcoマーカーが検出されず、キャリブレーションができません")
        continue

    # 🔹 キャリブレーション用のデータ数を制限
    if len(objpoints) > 100:
        objpoints = objpoints[:100]
        imgpoints = imgpoints[:100]

    assert len(objpoints) == len(imgpoints), f"❌ {video_path}: objpoints と imgpoints のサイズが一致しません"

    # ✅ 内部パラメータを計算
    print(f"⏳ {video_path}: キャリブレーションを実行中...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frame_shape[::-1], None, None)

    camera_matrices.append(mtx)
    distortion_coeffs.append(dist)
    rvecs_list.append(rvecs)
    tvecs_list.append(tvecs)

    print(f"✅ キャリブレーション成功: {video_path}")

# カメラパラメータを保存
np.savez("camera_calibration_params.npz",
         camera_matrices=camera_matrices,
         distortion_coeffs=distortion_coeffs,
         rvecs_list=rvecs_list,
         tvecs_list=tvecs_list)

print("✅ 全カメラのキャリブレーション完了！")
