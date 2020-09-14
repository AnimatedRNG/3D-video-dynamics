import numpy as np
import torch as torch
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.data_classes import PointCloud, LidarPointCloud
from pyquaternion import Quaternion
import os.path as osp
from functools import reduce

NUM_LIDAR_CHANNELS = 4  # XYZTR


def from_file(file_name: str):
    scan = np.fromfile(file_name, dtype=np.float32)
    points = scan.reshape((-1, 5))
    xyzr = points[:, : LidarPointCloud.nbr_dims()]
    ring = points[:, 4]
    xyzr[:, 3] = ring
    return LidarPointCloud(xyzr.T)


# very simple dataset for nuscenes pointcloud data
# based off of https://github.com/nutonomy/nuscenes-devkit/blob/274725ae1b3a2d921725016e3f4b383b8b218d3a/python-sdk/nuscenes/utils/data_classes.py#L55
class NuscenesPointCloudDataset(Dataset):
    def __init__(self, scene_id: int, min_distance: float = 1.0, skip_sweep: int = 1):
        super().__init__()

        self.nusc = NuScenes(
            version="v1.0-mini", dataroot="./data/sets/nuscenes", verbose=True
        )

        self.scene = self.nusc.scene[scene_id]
        first_sample_token = self.scene["first_sample_token"]
        self.first_sample = self.nusc.get("sample", first_sample_token)

        points = np.zeros((NUM_LIDAR_CHANNELS, 0), dtype=np.float32)
        num_samples = self.scene["nbr_samples"]
        all_pc = [LidarPointCloud(points) for _ in range(num_samples)]

        ref_sd_token = self.first_sample["data"]["LIDAR_TOP"]
        ref_sd_rec = self.nusc.get("sample_data", ref_sd_token)
        ref_pose_rec = self.nusc.get("ego_pose", ref_sd_rec["ego_pose_token"])
        ref_cs_rec = self.nusc.get(
            "calibrated_sensor", ref_sd_rec["calibrated_sensor_token"]
        )
        ref_time = 1e-6 * ref_sd_rec["timestamp"]

        # Homogeneous transform from ego car frame to reference frame.
        ref_from_car = transform_matrix(
            ref_cs_rec["translation"], Quaternion(ref_cs_rec["rotation"]), inverse=True
        )

        # Homogeneous transformation matrix from global to _current_ ego car frame.
        car_from_global = transform_matrix(
            ref_pose_rec["translation"],
            Quaternion(ref_pose_rec["rotation"]),
            inverse=True,
        )

        current_sample = self.first_sample

        self.pc = []
        self.ego = []

        for sample_id in range(1):
            sample_data_token = current_sample["data"]["LIDAR_TOP"]
            current_sd_rec = self.nusc.get("sample_data", sample_data_token)

            print("loading sample...", sample_id)

            while current_sd_rec["next"] != "":
                ego = LidarPointCloud(
                    np.array([[0.0], [0.0], [0.0], [1.0]], dtype=np.float32)
                )
                current_pc = from_file(
                    osp.join(self.nusc.dataroot, current_sd_rec["filename"])
                )
                current_pc.remove_close(min_distance)
                ring = current_pc.points[3].copy()

                # Get past pose.
                current_pose_rec = self.nusc.get(
                    "ego_pose", current_sd_rec["ego_pose_token"]
                )
                global_from_car = transform_matrix(
                    current_pose_rec["translation"],
                    Quaternion(current_pose_rec["rotation"]),
                    inverse=False,
                )

                # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
                current_cs_rec = self.nusc.get(
                    "calibrated_sensor", current_sd_rec["calibrated_sensor_token"]
                )
                car_from_current = transform_matrix(
                    current_cs_rec["translation"],
                    Quaternion(current_cs_rec["rotation"]),
                    inverse=False,
                )

                # Fuse four transformation matrices into one and perform transform.
                trans_matrix = reduce(
                    np.dot,
                    [ref_from_car, car_from_global, global_from_car, car_from_current],
                )
                current_pc.transform(trans_matrix)
                ego.transform(trans_matrix)

                time_lag = (
                    1e-6 * current_sd_rec["timestamp"] - ref_time
                )  # Positive difference.
                times = time_lag * np.ones((1, current_pc.nbr_points()))

                # replace the intensity with time
                current_pc.points[3] = times
                ring_pts = np.concatenate(
                    (current_pc.points, ring.reshape(1, -1)), axis=0
                )
                ring_pts = torch.from_numpy(ring_pts.transpose())
                rings_idx = [
                    (ring_pts[:, -1] == i).nonzero().squeeze(1).type(torch.LongTensor)
                    for i in range(int(ring_pts[:, -1].max().item()) + 1)
                ]
                all_idx = torch.cat(rings_idx)
                pc_sorted = ring_pts[all_idx]

                self.pc.append(pc_sorted.numpy())
                self.ego.append(ego.points.transpose())
                # print(sample_id, ego.points.transpose())
                print(pc_sorted[:100])

                for _ in range(skip_sweep):
                    if current_sd_rec["next"] == "":
                        break
                    current_sd_rec = self.nusc.get(
                        "sample_data", current_sd_rec["next"]
                    )

            if current_sample["next"] == "":
                break
            else:
                current_sample = self.nusc.get("sample", current_sample["next"])

    def __len__(self):
        return len(self.pc)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.pc[idx]).cuda(),
            torch.from_numpy(self.ego[idx]).cuda(),
        )


# example usage
if __name__ == "__main__":
    dset = NuscenesPointCloudDataset(0)
