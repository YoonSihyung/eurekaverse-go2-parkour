import imageio
import os
import gymnasium as gym
import numpy as np
import torch
from scipy.spatial.transform import Rotation


class MultiCamVideo(gym.Wrapper):
    def __init__(self, env, out_dir, cam_names:list, fps=30, length=float("inf"),
                 cam_specs:dict=None, sensor_key:str=None,
                 ego_sensor_key:str=None, ego_cam_specs:dict=None, ego_suffix:str="_ego"):
        """cam_specs: optional {cam_name: env_index} for a single batched viz sensor
        (sensor_key); when None, each cam_name is its own single-prim sensor.
        ego_sensor_key/ego_cam_specs: optional second batched sensor (e.g. an onboard
        egocentric RGB camera) recorded in parallel to {cam_name}{ego_suffix}.mp4."""
        super().__init__(env)
        self.out_dir, self.fps = out_dir, fps
        self.len = length
        os.makedirs(out_dir, exist_ok=True)
        self.cam_names = cam_names
        self.cam_specs, self.sensor_key = cam_specs, sensor_key
        self.ego_sensor_key, self.ego_cam_specs, self.ego_suffix = ego_sensor_key, ego_cam_specs, ego_suffix
        self.writers = {
            cam_name: imageio.get_writer(f"{self.out_dir}/{cam_name}.mp4", fps=self.fps)
            for cam_name in cam_names
        }
        if self.ego_sensor_key:
            self.ego_writers = {
                cam_name: imageio.get_writer(f"{self.out_dir}/{cam_name}{self.ego_suffix}.mp4", fps=self.fps)
                for cam_name in cam_names
            }
        self.frame = 0

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)

        if self.frame < self.len:
            batched = self.env.scene.sensors[self.sensor_key].data.output["rgb"] if self.cam_specs else None
            ego_batched = self.env.scene.sensors[self.ego_sensor_key].data.output["rgb"] if self.ego_sensor_key else None
            for cam_name in self.cam_names:
                if self.cam_specs:
                    image = batched[self.cam_specs[cam_name]]
                else:
                    image = self.env.scene.sensors[cam_name].data.output["rgb"].squeeze(0)
                assert len(image.shape) == 3, f"Expected image shape to be 3D, got {image.shape}"
                self.writers[cam_name].append_data(image.cpu().numpy().astype("uint8"))
                if ego_batched is not None:
                    ego_image = ego_batched[self.ego_cam_specs[cam_name]]
                    # RGBA (TiledCamera) -> RGB
                    ego_image = ego_image[..., :3]
                    self.ego_writers[cam_name].append_data(ego_image.cpu().numpy().astype("uint8"))

        self.frame += 1
        return obs, r, term, trunc, info

    def close(self):
        # Close all video writers and then the underlying environment
        for writer in self.writers.values():
            writer.close()
        if self.ego_sensor_key:
            for writer in self.ego_writers.values():
                writer.close()
        super().close()


def get_camera_coords(col_idx, row_idx, env_origin, terrain_length=18.0):
    """
    Camera pose for one terrain cell, computed from the cell's actual env origin.

    Three-quarter view from ahead-left-above the robot's direction of travel
    (robot moves +x, its left is +y): terrain structure and robot motion are both
    readable. Steep enough (~45°) that obstacles in the neighboring column
    cannot occlude the course. Rotation is returned as (x, y, z, w) matching
    Isaac Lab 3.0's quaternion order, camera "world" convention (x-fwd, z-up).
    """
    ox, oy, oz = float(env_origin[0]), float(env_origin[1]), float(env_origin[2])

    pos = (ox + 10.0, oy + 6.0, oz + 6.0)
    target = (ox + 4.0, oy, oz + 0.3)

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
