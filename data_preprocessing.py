import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
from scipy.spatial.transform import Rotation

class SLAMDataPreprocessor:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.image_path = self.dataset_path / "images"
        self.depth_path = self.dataset_path / "depth"
        self.trajectory_file = self.dataset_path / "trajectory.npy"
        self.camera_info_file = self.dataset_path / "camera_info.npy"

        self.output_path = self.dataset_path / "preprocessed"
        self.output_path.mkdir(exist_ok=True)

        self.load_dataset()

    def load_dataset(self):
        if not self.trajectory_file.exists():
            raise FileNotFoundError(f"Trajectory file not found: {self.trajectory_file}")

        if not self.camera_info_file.exists():
            raise FileNotFoundError(f"Camera info file not found: {self.camera_info_file}")

        self.trajectory_data = np.load(str(self.trajectory_file), allow_pickle=True)
        self.camera_info = np.load(str(self.camera_info_file), allow_pickle=True).item()

        print(f"Loaded dataset with {len(self.trajectory_data)} frames")
        print(f"Camera info: {self.camera_info}")

    def convert_to_tum_format(self):
        tum_format_path = self.output_path / "tum_format"
        tum_format_path.mkdir(exist_ok=True)

        rgb_path = tum_format_path / "rgb"
        rgb_path.mkdir(exist_ok=True)

        depth_path = tum_format_path / "depth"
        depth_path.mkdir(exist_ok=True)

        trajectory_file = tum_format_path / "groundtruth.txt"

        with open(trajectory_file, 'w') as f:
            f.write("# timestamp tx ty tz qx qy qz qw\n")

            for frame in self.trajectory_data:
                timestamp = frame['timestamp']
                pos = frame['gt_position']
                ori = frame['gt_orientation']

                f.write(f"{timestamp:.6f} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} "
                        f"{ori[1]:.6f} {ori[2]:.6f} {ori[3]:.6f} {ori[0]:.6f}\n")

        associations_file = tum_format_path / "associations.txt"

        with open(associations_file, 'w') as f:
            f.write("# rgb_timestamp rgb_file depth_timestamp depth_file\n")

            for frame in self.trajectory_data:
                if 'frame_id' not in frame:
                    continue

                frame_id = frame['frame_id']
                timestamp = frame['timestamp']

                rgb_file = f"rgb/{frame_id:06d}.png"
                depth_file = f"depth/{frame_id:06d}.png"

                f.write(f"{timestamp:.6f} {rgb_file} {timestamp:.6f} {depth_file}\n")

                src_rgb = self.image_path / f"{frame_id:06d}.png"
                dst_rgb = rgb_path / f"{frame_id:06d}.png"

                if src_rgb.exists():
                    img = cv2.imread(str(src_rgb))
                    cv2.imwrite(str(dst_rgb), img)

                src_depth = self.depth_path / f"{frame_id:06d}.npy"
                dst_depth = depth_path / f"{frame_id:06d}.png"

                if src_depth.exists():
                    depth = np.load(str(src_depth))

                    if len(depth.shape) == 3:
                        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)

                    depth_scaled = (depth * 5000).astype(np.uint16)
                    cv2.imwrite(str(dst_depth), depth_scaled)

        calib_file = tum_format_path / "calibration.txt"

        with open(calib_file, 'w') as f:
            f.write(f"# Camera calibration parameters\n")
            f.write(f"fx={self.camera_info['fx']}\n")
            f.write(f"fy={self.camera_info['fy']}\n")
            f.write(f"cx={self.camera_info['cx']}\n")
            f.write(f"cy={self.camera_info['cy']}\n")
            f.write(f"width={self.camera_info['width']}\n")
            f.write(f"height={self.camera_info['height']}\n")

        return tum_format_path

    def convert_to_euroc_format(self):
        euroc_path = self.output_path / "euroc_format"
        euroc_path.mkdir(exist_ok=True)

        mav_path = euroc_path / "mav0"
        mav_path.mkdir(exist_ok=True)

        cam0_path = mav_path / "cam0"
        cam0_path.mkdir(exist_ok=True)
        cam0_data_path = cam0_path / "data"
        cam0_data_path.mkdir(exist_ok=True)

        imu0_path = mav_path / "imu0"
        imu0_path.mkdir(exist_ok=True)

        cam0_calib = {
            "camera_model": "pinhole",
            "intrinsics": [
                self.camera_info['fx'],
                self.camera_info['fy'],
                self.camera_info['cx'],
                self.camera_info['cy']
            ],
            "distortion_model": "radtan",
            "distortion_coefficients": [0.0, 0.0, 0.0, 0.0],
            "resolution": [
                self.camera_info['width'],
                self.camera_info['height']
            ],
            "timeshift_cam_imu": 0.0
        }

        with open(cam0_path / "sensor.yaml", 'w') as f:
            yaml_str = ""
            for key, value in cam0_calib.items():
                yaml_str += f"{key}: {value}\n"
            f.write(yaml_str)

        cam0_timestamps = []

        for frame in self.trajectory_data:
            if 'frame_id' not in frame:
                continue

            frame_id = frame['frame_id']
            timestamp = frame['timestamp']
            timestamp_ns = int(timestamp * 1e9)

            cam0_timestamps.append([timestamp_ns, frame_id])

            src_rgb = self.image_path / f"{frame_id:06d}.png"
            dst_rgb = cam0_data_path / f"{timestamp_ns}.png"

            if src_rgb.exists():
                img = cv2.imread(str(src_rgb))
                cv2.imwrite(str(dst_rgb), img)

        df_cam = pd.DataFrame(cam0_timestamps, columns=['timestamp', 'frame_id'])
        df_cam.to_csv(cam0_path / "data.csv", index=False)

        imu_data = []

        for frame in self.trajectory_data:
            if 'angular_velocity' not in frame or 'linear_acceleration' not in frame:
                continue

            timestamp = frame['timestamp']
            timestamp_ns = int(timestamp * 1e9)

            angular_vel = frame['angular_velocity']
            linear_acc = frame['linear_acceleration']

            imu_data.append([
                timestamp_ns,
                angular_vel[0], angular_vel[1], angular_vel[2],
                linear_acc[0], linear_acc[1], linear_acc[2]
            ])

        df_imu = pd.DataFrame(
            imu_data,
            columns=[
                'timestamp',
                'omega_x', 'omega_y', 'omega_z',
                'alpha_x', 'alpha_y', 'alpha_z'
            ]
        )
        df_imu.to_csv(imu0_path / "data.csv", index=False)

        imu_calib = {
            "accelerometer_noise_density": 0.01,
            "accelerometer_random_walk": 0.0002,
            "gyroscope_noise_density": 0.005,
            "gyroscope_random_walk": 0.0001
        }

        with open(imu0_path / "sensor.yaml", 'w') as f:
            yaml_str = ""
            for key, value in imu_calib.items():
                yaml_str += f"{key}: {value}\n"
            f.write(yaml_str)


        gt_path = mav_path / "state_groundtruth_estimate0"
        gt_path.mkdir(exist_ok=True)

        gt_data = []

        for frame in self.trajectory_data:
            timestamp = frame['timestamp']
            timestamp_ns = int(timestamp * 1e9)

            pos = frame['gt_position']
            ori = frame['gt_orientation']

            gt_data.append([
                timestamp_ns,
                pos[0], pos[1], pos[2],
                ori[1], ori[2], ori[3], ori[0],
                0, 0, 0,
                0, 0, 0,
                0, 0, 0,
            ])

        df_gt = pd.DataFrame(
            gt_data,
            columns=[
                'timestamp',
                'p_RS_R_x', 'p_RS_R_y', 'p_RS_R_z',
                'q_RS_x', 'q_RS_y', 'q_RS_z', 'q_RS_w',
                'v_RS_R_x', 'v_RS_R_y', 'v_RS_R_z',
                'b_a_x', 'b_a_y', 'b_a_z',
                'b_g_x', 'b_g_y', 'b_g_z'
            ]
        )
        df_gt.to_csv(gt_path / "data.csv", index=False)

        return euroc_path

    def convert_to_kitti_format(self):
        kitti_path = self.output_path / "kitti_format"
        kitti_path.mkdir(exist_ok=True)

        image_00_path = kitti_path / "image_00" / "data"
        image_00_path.mkdir(parents=True, exist_ok=True)

        times_file = kitti_path / "times.txt"

        with open(times_file, 'w') as f:
            for frame in self.trajectory_data:
                if 'frame_id' not in frame:
                    continue

                timestamp = frame['timestamp']
                f.write(f"{timestamp:.6f}\n")

        calib_file = kitti_path / "calib.txt"

        with open(calib_file, 'w') as f:
            P0 = np.zeros((3, 4))
            P0[0, 0] = self.camera_info['fx']
            P0[1, 1] = self.camera_info['fy']
            P0[0, 2] = self.camera_info['cx']
            P0[1, 2] = self.camera_info['cy']
            P0[2, 2] = 1.0

            P0_str = ' '.join(map(str, P0.flatten()))
            f.write(f"P0: {P0_str}\n")

        for frame in self.trajectory_data:
            if 'frame_id' not in frame:
                continue

            frame_id = frame['frame_id']

            src_rgb = self.image_path / f"{frame_id:06d}.png"
            dst_rgb = image_00_path / f"{frame_id:010d}.png"

            if src_rgb.exists():
                img = cv2.imread(str(src_rgb))
                cv2.imwrite(str(dst_rgb), img)

        poses_file = kitti_path / "poses.txt"

        with open(poses_file, 'w') as f:
            for frame in self.trajectory_data:
                if 'frame_id' not in frame:
                    continue

                pos = frame['gt_position']
                ori = frame['gt_orientation']  # [w, x, y, z] in AirSim

                quat = [ori[0], ori[1], ori[2], ori[3]]  # [w, x, y, z]
                rot = Rotation.from_quat([ori[1], ori[2], ori[3], ori[0]]).as_matrix()  # [x, y, z, w]

                transform = np.eye(4)
                transform[:3, :3] = rot
                transform[:3, 3] = pos

                transform_flat = transform[:3, :].flatten()
                transform_str = ' '.join([f"{v:.6f}" for v in transform_flat])

                f.write(f"{transform_str}\n")

        return kitti_path

    def create_ml_training_dataset(self, output_size=(224, 224), patches_per_image=10, patch_size=(64, 64)):
        ml_dataset_path = self.output_path / "ml_dataset"
        ml_dataset_path.mkdir(exist_ok=True)

        normal_path = ml_dataset_path / "normal"
        normal_path.mkdir(exist_ok=True)

        low_light_path = ml_dataset_path / "low_light"
        low_light_path.mkdir(exist_ok=True)

        shadow_path = ml_dataset_path / "shadow"
        shadow_path.mkdir(exist_ok=True)

        for frame in self.trajectory_data:
            if 'frame_id' not in frame:
                continue

            frame_id = frame['frame_id']

            src_rgb = self.image_path / f"{frame_id:06d}.png"

            if not src_rgb.exists():
                continue

            img = cv2.imread(str(src_rgb))

            img_resized = cv2.resize(img, output_size)

            cv2.imwrite(str(normal_path / f"{frame_id:06d}.png"), img_resized)

            low_light = (img_resized * 0.3).astype(np.uint8)
            cv2.imwrite(str(low_light_path / f"{frame_id:06d}.png"), low_light)

            shadow = img_resized.copy()
            shadow[:, :shadow.shape[1]//2] = (shadow[:, :shadow.shape[1]//2] * 0.4).astype(np.uint8)
            cv2.imwrite(str(shadow_path / f"{frame_id:06d}.png"), shadow)

            patches_dir = ml_dataset_path / "patches" / f"{frame_id:06d}"
            patches_dir.mkdir(parents=True, exist_ok=True)

            h, w = img.shape[:2]

            for i in range(patches_per_image):
                x = np.random.randint(0, w - patch_size[0])
                y = np.random.randint(0, h - patch_size[1])

                patch = img[y:y+patch_size[1], x:x+patch_size[0]]

                cv2.imwrite(str(patches_dir / f"orig_{i:02d}.png"), patch)

                low_light_patch = (patch * 0.3).astype(np.uint8)
                cv2.imwrite(str(patches_dir / f"low_{i:02d}.png"), low_light_patch)

                shadow_patch = patch.copy()
                shadow_patch[:, :shadow_patch.shape[1]//2] = (shadow_patch[:, :shadow_patch.shape[1]//2] * 0.4).astype(np.uint8)
                cv2.imwrite(str(patches_dir / f"shadow_{i:02d}.png"), shadow_patch)

        return ml_dataset_path

    def process_dataset(self, formats=["tum", "euroc", "kitti"], create_ml_dataset=True):
        output_paths = {}

        if "tum" in formats:
            output_paths["tum"] = self.convert_to_tum_format()

        if "euroc" in formats:
            output_paths["euroc"] = self.convert_to_euroc_format()

        if "kitti" in formats:
            output_paths["kitti"] = self.convert_to_kitti_format()

        if create_ml_dataset:
            output_paths["ml_dataset"] = self.create_ml_training_dataset()

        return output_paths

if __name__ == "__main__":
    dataset_path = "./data/urban_canyon_dataset"

    preprocessor = SLAMDataPreprocessor(dataset_path)
    output_paths = preprocessor.process_dataset(formats=["tum", "euroc", "kitti"])

    print("Preprocessing completed. Output paths:")
    for format_name, path in output_paths.items():
        print(f"  {format_name}: {path}")
