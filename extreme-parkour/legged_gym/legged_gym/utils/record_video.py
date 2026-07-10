import imageio
import os
import gymnasium as gym
import numpy as np
import torch
from scipy.spatial.transform import Rotation


class MultiCamVideo(gym.Wrapper):
    def __init__(self, env, out_dir, cam_names:list, fps=30, length=float("inf")):
        super().__init__(env)
        self.out_dir, self.fps = out_dir, fps
        self.len = length
        os.makedirs(out_dir, exist_ok=True)
        self.cam_names = cam_names
        self.writers = {
            cam_name: imageio.get_writer(f"{self.out_dir}/{cam_name}.mp4", fps=self.fps)
            for cam_name in cam_names
        }
        self.frame = 0

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)

        if self.frame < self.len:
            for cam_name in self.cam_names:
                image = self.env.scene.sensors[cam_name].data.output["rgb"].squeeze(0)
                assert len(image.shape) == 3, f"Expected image shape to be 3D, got {image.shape}"
                self.writers[cam_name].append_data(image.cpu().numpy().astype("uint8"))

        self.frame += 1
        return obs, r, term, trunc, info

    def close(self):
        # Close all video writers and then the underlying environment
        for writer in self.writers.values():
            writer.close()
        super().close()


def get_camera_coords(col_idx, row_idx, env_origin, terrain_length=18.0, cam_height=3.2, back_offset=-3.5):
    """
    Camera pose for one terrain cell, computed from the cell's actual env origin.

    Behind-the-spawn view looking down the course (+x): stays inside the cell's own
    4m-wide corridor, so tall obstacles in neighboring columns can never occlude it.
    Robots walk away from the camera with obstacles readable in depth.
    Rotation is returned as (x, y, z, w) matching Isaac Lab 3.0's quaternion order,
    for a camera in "world" convention (x-forward, y-left, z-up).
    """
    ox, oy, oz = float(env_origin[0]), float(env_origin[1]), float(env_origin[2])

    pos = (ox + back_offset, oy, oz + cam_height)
    target = (ox + 6.5, oy, oz + 0.3)

    f = np.array(target) - np.array(pos)
    f = f / np.linalg.norm(f)                    # camera x-axis (forward)
    left = np.cross(np.array([0.0, 0.0, 1.0]), f)
    left = left / np.linalg.norm(left)           # camera y-axis (left)
    up = np.cross(f, left)                       # camera z-axis (up)
    rot_mat = np.column_stack([f, left, up])
    quat_xyzw = Rotation.from_matrix(rot_mat).as_quat()

    return {
        "position": pos,
        "rotation": tuple(quat_xyzw),
    }
