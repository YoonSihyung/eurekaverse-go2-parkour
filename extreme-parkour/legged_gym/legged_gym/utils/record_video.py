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


def get_camera_coords(col_idx, row_idx, env_origin, terrain_length=18.0, cam_height=8.5, side_offset=-7.5):
    # Steeper, closer side view: crosses at most one neighboring 4m column, so tall
    # obstacles in adjacent courses can't occlude the target course.
    """
    Camera pose for one terrain cell, computed from the cell's actual env origin.

    Elevated side view covering the whole course: positioned to the -y side of the
    cell, centered along the course (+x) direction, looking at the course center.
    Rotation is returned as (x, y, z, w) matching Isaac Lab 3.0's quaternion order,
    for a camera in "world" convention (x-forward, y-left, z-up).
    """
    ox, oy, oz = float(env_origin[0]), float(env_origin[1]), float(env_origin[2])
    # Aim at the spawn-to-midcourse stretch (x ∈ [origin-1, origin+10]) — on hard
    # difficulties robots rarely pass midcourse, so centering on the full course
    # leaves them out of frame.
    course_center_x = ox + 4.5

    pos = (course_center_x, oy + side_offset, oz + cam_height)
    target = (course_center_x, oy, oz + 0.5)

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
