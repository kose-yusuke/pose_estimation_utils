import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# ✅ 3Dキーポイントをロード
with open("keypoints_3d.json", "r") as f:
    keypoints_3d = json.load(f)

# ✅ 有効なフレームのみ取得（空リストがないもの）
valid_frames = [frame for frame in keypoints_3d if len(frame) == 17 and all(len(kp) == 3 for kp in frame)]
print(f"📌 有効なフレーム数: {len(valid_frames)} / {len(keypoints_3d)}")

if len(valid_frames) == 0:
    print("🚨 可視化できるデータがありません！")
    exit()

# ✅ 3Dプロットのセットアップ
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# 🎥 **初期視点**
elev = 20  # 上下回転角度
azim = 60  # 水平回転角度
ax.view_init(elev, azim)

# 🎯 **座標範囲を固定（チカチカ防止）**
ax.set_xlim(-40, 40)
ax.set_ylim(-40, 40)
ax.set_zlim(-20, 20)


# 🎮 **キーボード操作で視点変更**
def on_key(event):
    global elev, azim
    step = 10  # 回転のステップサイズ

    if event.key == "w":
        elev += step  # 上に回転
    elif event.key == "s":
        elev -= step  # 下に回転
    elif event.key == "a":
        azim -= step  # 左に回転
    elif event.key == "d":
        azim += step  # 右に回転
    elif event.key == "q":
        ax.set_xlim(ax.get_xlim()[0] * 1.1, ax.get_xlim()[1] * 1.1)  # 拡大
        ax.set_ylim(ax.get_ylim()[0] * 1.1, ax.get_ylim()[1] * 1.1)
        ax.set_zlim(ax.get_zlim()[0] * 1.1, ax.get_zlim()[1] * 1.1)
    elif event.key == "e":
        ax.set_xlim(ax.get_xlim()[0] * 0.9, ax.get_xlim()[1] * 0.9)  # 縮小
        ax.set_ylim(ax.get_ylim()[0] * 0.9, ax.get_ylim()[1] * 0.9)
        ax.set_zlim(ax.get_zlim()[0] * 0.9, ax.get_zlim()[1] * 0.9)

    ax.view_init(elev, azim)
    plt.draw()


fig.canvas.mpl_connect("key_press_event", on_key)

# ✅ フレームごとに 3D キーポイントを更新
for frame_idx, frame in enumerate(valid_frames):
    if not frame or any(len(kp) != 3 for kp in frame):
        print(f"⚠️ フレーム {frame_idx} は無効なデータのためスキップ")
        continue

    ax.clear()  # 前フレームのデータをクリア
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Frame {frame_idx}")

    # ✅ **座標範囲をリセット（視点を固定）**
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)

    # ✅ 3D キーポイントを取得
    try:
        x, y, z = zip(*frame)  # 各キーポイントの x, y, z
        ax.scatter(x, y, z, c="r", marker="o")
        ax.view_init(elev, azim)  # 現在の視点を適用
    except ValueError as e:
        print(f"⚠️ フレーム {frame_idx} でエラー: {e}")
        continue

    plt.pause(0.2)  # 50ms ごとに更新

plt.show()
