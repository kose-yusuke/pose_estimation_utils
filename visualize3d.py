import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


with open("keypoints_3d.json", "r") as f:
    keypoints_3d = json.load(f)


valid_frames = [frame for frame in keypoints_3d if len(frame) == 17 and all(len(kp) == 3 for kp in frame)]
print(f"📌 valid flame number: {len(valid_frames)} / {len(keypoints_3d)}")

if len(valid_frames) == 0:
    print("🚨 no data available to visualize")
    exit()


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

elev = 20
azim = 60
ax.view_init(elev, azim)

ax.set_xlim(-40, 40)
ax.set_ylim(-40, 40)
ax.set_zlim(-20, 20)

def on_key(event):
    global elev, azim
    step = 10

    if event.key == "w":
        elev += step
    elif event.key == "s":
        elev -= step
    elif event.key == "a":
        azim -= step
    elif event.key == "d":
        azim += step
    elif event.key == "q":
        ax.set_xlim(ax.get_xlim()[0] * 1.1, ax.get_xlim()[1] * 1.1)
        ax.set_ylim(ax.get_ylim()[0] * 1.1, ax.get_ylim()[1] * 1.1)
        ax.set_zlim(ax.get_zlim()[0] * 1.1, ax.get_zlim()[1] * 1.1)
    elif event.key == "e":
        ax.set_xlim(ax.get_xlim()[0] * 0.9, ax.get_xlim()[1] * 0.9)
        ax.set_ylim(ax.get_ylim()[0] * 0.9, ax.get_ylim()[1] * 0.9)
        ax.set_zlim(ax.get_zlim()[0] * 0.9, ax.get_zlim()[1] * 0.9)

    ax.view_init(elev, azim)
    plt.draw()


fig.canvas.mpl_connect("key_press_event", on_key)


for frame_idx, frame in enumerate(valid_frames):
    if not frame or any(len(kp) != 3 for kp in frame):
        print(f"⚠️ frame {frame_idx} is invalid so we skip it")
        continue

    ax.clear()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Frame {frame_idx}")


    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)

    try:
        x, y, z = zip(*frame)
        ax.scatter(x, y, z, c="r", marker="o")
        ax.view_init(elev, azim)
    except ValueError as e:
        print(f"⚠️ frame {frame_idx} error: {e}")
        continue

    plt.pause(0.2)

plt.show()
