import os
import sys
import subprocess
import numpy as np
import cv2
import time
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
from scipy.spatial.transform import Rotation
import json
import glob

class SLAMEvaluator:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.results_path = self.dataset_path / "results"
        self.results_path.mkdir(exist_ok=True)

    def run_orb_slam3(self, dataset_format="tum", sensor_type="mono"):
        orb_slam_path = Path("./ORB_SLAM3")

        if not orb_slam_path.exists():
            raise FileNotFoundError(f"ORB_SLAM3 not found at {orb_slam_path}")

        # Ensure results directory exists
        result_dir = self.results_path / "orb_slam3" / dataset_format / sensor_type
        result_dir.mkdir(parents=True, exist_ok=True)

        # Prepare command and paths based on dataset format
        if dataset_format == "tum":
            dataset_dir = self.dataset_path / "preprocessed" / "tum_format"
            vocab_path = orb_slam_path / "Vocabulary" / "ORBvoc.txt"

            if sensor_type == "mono":
                config_path = orb_slam_path / "Examples" / "Monocular" / "TUM1.yaml"
                cmd = [
                    str(orb_slam_path / "Examples" / "Monocular" / "mono_tum"),
                    str(vocab_path),
                    str(config_path),
                    str(dataset_dir),
                    str(dataset_dir / "associations.txt")
                ]
            elif sensor_type == "stereo":
                config_path = orb_slam_path / "Examples" / "Stereo" / "TUM1.yaml"
                cmd = [
                    str(orb_slam_path / "Examples" / "Stereo" / "stereo_tum"),
                    str(vocab_path),
                    str(config_path),
                    str(dataset_dir),
                    str(dataset_dir / "associations.txt")
                ]
            elif sensor_type == "rgbd":
                config_path = orb_slam_path / "Examples" / "RGB-D" / "TUM1.yaml"
                cmd = [
                    str(orb_slam_path / "Examples" / "RGB-D" / "rgbd_tum"),
                    str(vocab_path),
                    str(config_path),
                    str(dataset_dir),
                    str(dataset_dir / "associations.txt")
                ]
            else:
                raise ValueError(f"Unsupported sensor type: {sensor_type}")

        elif dataset_format == "euroc":
            dataset_dir = self.dataset_path / "preprocessed" / "euroc_format"
            vocab_path = orb_slam_path / "Vocabulary" / "ORBvoc.txt"

            if sensor_type == "mono":
                config_path = orb_slam_path / "Examples" / "Monocular" / "EuRoC.yaml"
                cmd = [
                    str(orb_slam_path / "Examples" / "Monocular" / "mono_euroc"),
                    str(vocab_path),
                    str(config_path),
                    str(dataset_dir)
                ]
            elif sensor_type == "stereo":
                config_path = orb_slam_path / "Examples" / "Stereo" / "EuRoC.yaml"
                cmd = [
                    str(orb_slam_path / "Examples" / "Stereo" / "stereo_euroc"),
                    str(vocab_path),
                    str(config_path),
                    str(dataset_dir)
                ]
            else:
                raise ValueError(f"Unsupported sensor type: {sensor_type} for EuRoC format")

        elif dataset_format == "kitti":
            dataset_dir = self.dataset_path / "preprocessed" / "kitti_format"
            vocab_path = orb_slam_path / "Vocabulary" / "ORBvoc.txt"

            if sensor_type == "mono":
                config_path = orb_slam_path / "Examples" / "Monocular" / "KITTI00-02.yaml"
                cmd = [
                    str(orb_slam_path / "Examples" / "Monocular" / "mono_kitti"),
                    str(vocab_path),
                    str(config_path),
                    str(dataset_dir)
                ]
            elif sensor_type == "stereo":
                config_path = orb_slam_path / "Examples" / "Stereo" / "KITTI00-02.yaml"
                cmd = [
                    str(orb_slam_path / "Examples" / "Stereo" / "stereo_kitti"),
                    str(vocab_path),
                    str(config_path),
                    str(dataset_dir)
                ]
            else:
                raise ValueError(f"Unsupported sensor type: {sensor_type} for KITTI format")
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_format}")

        # Run ORB-SLAM3
        print(f"Running ORB-SLAM3 with command: {' '.join(cmd)}")

        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(orb_slam_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )

            stdout, stderr = process.communicate()

            with open(result_dir / "stdout.txt", "w") as f:
                f.write(stdout)

            with open(result_dir / "stderr.txt", "w") as f:
                f.write(stderr)

            # Copy the trajectory result file
            trajectory_file = orb_slam_path / "CameraTrajectory.txt"
            if trajectory_file.exists():
                shutil.copy(
                    trajectory_file,
                    result_dir / "CameraTrajectory.txt"
                )

            # Copy the keyframes file
            keyframes_file = orb_slam_path / "KeyFrameTrajectory.txt"
            if keyframes_file.exists():
                shutil.copy(
                    keyframes_file,
                    result_dir / "KeyFrameTrajectory.txt"
                )

            return result_dir

        except Exception as e:
            print(f"Error running ORB-SLAM3: {e}")
            return None

    def run_dso(self, dataset_format="tum"):
        dso_path = Path("./dso")

        if not dso_path.exists():
            raise FileNotFoundError(f"DSO not found at {dso_path}")

        # Ensure results directory exists
        result_dir = self.results_path / "dso" / dataset_format
        result_dir.mkdir(parents=True, exist_ok=True)

        # Prepare command and paths based on dataset format
        if dataset_format == "tum":
            dataset_dir = self.dataset_path / "preprocessed" / "tum_format"

            # DSO requires images in a specific format
            cmd = [
                str(dso_path / "build" / "bin" / "dso_dataset"),
                "files=" + str(dataset_dir / "images"),
                "calib=" + str(dataset_dir / "camera.txt"),
                "gamma=1",
                "vignette=1",
                "preset=0",
                "mode=0",
                "result=" + str(result_dir / "result.txt")
            ]
        elif dataset_format == "euroc":
            dataset_dir = self.dataset_path / "preprocessed" / "euroc_format"

            # DSO requires a different format for EuRoC
            cmd = [
                str(dso_path / "build" / "bin" / "dso_dataset"),
                "files=" + str(dataset_dir / "mav0/cam0/data"),
                "calib=" + str(dataset_dir / "mav0/cam0/camera.txt"),
                "gamma=1",
                "vignette=1",
                "preset=0",
                "mode=0",
                "result=" + str(result_dir / "result.txt")
            ]
        else:
            raise ValueError(f"Unsupported dataset format for DSO: {dataset_format}")

        # Run DSO
        print(f"Running DSO with command: {' '.join(cmd)}")

        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(dso_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )

            stdout, stderr = process.communicate()

            with open(result_dir / "stdout.txt", "w") as f:
                f.write(stdout)

            with open(result_dir / "stderr.txt", "w") as f:
                f.write(stderr)

            return result_dir

        except Exception as e:
            print(f"Error running DSO: {e}")
            return None

    def run_svo(self, dataset_format="tum"):
        svo_path = Path("./svo")

        if not svo_path.exists():
            raise FileNotFoundError(f"SVO not found at {svo_path}")

        # Ensure results directory exists
        result_dir = self.results_path / "svo" / dataset_format
        result_dir.mkdir(parents=True, exist_ok=True)

        # Prepare command and paths based on dataset format
        if dataset_format == "tum":
            dataset_dir = self.dataset_path / "preprocessed" / "tum_format"

            cmd = [
                str(svo_path / "bin" / "test_pipeline"),
                "-datasetdir=" + str(dataset_dir),
                "-cam_calib=" + str(dataset_dir / "calibration.txt"),
                "-logfile=" + str(result_dir / "log.txt"),
                "-traj_file=" + str(result_dir / "trajectory.txt")
            ]
        elif dataset_format == "euroc":
            dataset_dir = self.dataset_path / "preprocessed" / "euroc_format"

            cmd = [
                str(svo_path / "bin" / "test_pipeline"),
                "-datasetdir=" + str(dataset_dir),
                "-cam_calib=" + str(dataset_dir / "mav0/cam0/camera.txt"),
                "-logfile=" + str(result_dir / "log.txt"),
                "-traj_file=" + str(result_dir / "trajectory.txt")
            ]
        else:
            raise ValueError(f"Unsupported dataset format for SVO: {dataset_format}")

        # Run SVO
        print(f"Running SVO with command: {' '.join(cmd)}")

        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(svo_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )

            stdout, stderr = process.communicate()

            with open(result_dir / "stdout.txt", "w") as f:
                f.write(stdout)

            with open(result_dir / "stderr.txt", "w") as f:
                f.write(stderr)

            return result_dir

        except Exception as e:
            print(f"Error running SVO: {e}")
            return None

    def calculate_ate(self, gt_file, est_file, max_difference=0.02):
        """
        Calculate Absolute Trajectory Error (ATE)
        """
        # Load ground truth trajectory
        gt_data = np.loadtxt(gt_file)
        est_data = np.loadtxt(est_file)

        # Extract timestamps, positions and orientations
        gt_timestamps = gt_data[:, 0]
        gt_positions = gt_data[:, 1:4]
        gt_quaternions = gt_data[:, 4:8]  # [qx, qy, qz, qw]

        est_timestamps = est_data[:, 0]
        est_positions = est_data[:, 1:4]
        est_quaternions = est_data[:, 4:8]  # [qx, qy, qz, qw]

        # Associate data based on timestamps
        matched_gt_indices = []
        matched_est_indices = []

        for i, gt_time in enumerate(gt_timestamps):
            best_diff = float('inf')
            best_idx = -1

            for j, est_time in enumerate(est_timestamps):
                diff = abs(gt_time - est_time)

                if diff < best_diff and diff < max_difference:
                    best_diff = diff
                    best_idx = j

            if best_idx != -1:
                matched_gt_indices.append(i)
                matched_est_indices.append(best_idx)

        # Extract matched poses
        matched_gt_positions = gt_positions[matched_gt_indices]
        matched_gt_quaternions = gt_quaternions[matched_gt_indices]

        matched_est_positions = est_positions[matched_est_indices]
        matched_est_quaternions = est_quaternions[matched_est_indices]

        # Calculate error
        position_errors = matched_gt_positions - matched_est_positions
        position_error_norms = np.linalg.norm(position_errors, axis=1)

        # Calculate statistics
        rmse = np.sqrt(np.mean(np.square(position_error_norms)))
        mean = np.mean(position_error_norms)
        median = np.median(position_error_norms)
        std = np.std(position_error_norms)
        min_error = np.min(position_error_norms)
        max_error = np.max(position_error_norms)

        results = {
            "rmse": rmse,
            "mean": mean,
            "median": median,
            "std": std,
            "min": min_error,
            "max": max_error,
            "num_matches": len(matched_gt_indices)
        }

        return results, position_error_norms

    def calculate_rpe(self, gt_file, est_file, max_difference=0.02, delta=1.0):
        """
        Calculate Relative Pose Error (RPE)
        """
        # Load ground truth trajectory
        gt_data = np.loadtxt(gt_file)
        est_data = np.loadtxt(est_file)

        # Extract timestamps, positions and orientations
        gt_timestamps = gt_data[:, 0]
        gt_positions = gt_data[:, 1:4]
        gt_quaternions = gt_data[:, 4:8]  # [qx, qy, qz, qw]

        est_timestamps = est_data[:, 0]
        est_positions = est_data[:, 1:4]
        est_quaternions = est_data[:, 4:8]  # [qx, qy, qz, qw]

        # Associate data based on timestamps
        matched_gt_indices = []
        matched_est_indices = []

        for i, gt_time in enumerate(gt_timestamps):
            best_diff = float('inf')
            best_idx = -1

            for j, est_time in enumerate(est_timestamps):
                diff = abs(gt_time - est_time)

                if diff < best_diff and diff < max_difference:
                    best_diff = diff
                    best_idx = j

            if best_idx != -1:
                matched_gt_indices.append(i)
                matched_est_indices.append(best_idx)

        # Extract matched poses
        matched_gt_timestamps = gt_timestamps[matched_gt_indices]
        matched_gt_positions = gt_positions[matched_gt_indices]
        matched_gt_quaternions = gt_quaternions[matched_gt_indices]

        matched_est_timestamps = est_timestamps[matched_est_indices]
        matched_est_positions = est_positions[matched_est_indices]
        matched_est_quaternions = est_quaternions[matched_est_indices]

        # Calculate relative poses
        translation_errors = []
        rotation_errors = []

        for i in range(len(matched_gt_timestamps) - 1):
            j = i + 1

            # Skip if time difference is too small
            if matched_gt_timestamps[j] - matched_gt_timestamps[i] < delta:
                continue

            # Ground truth relative transformation
            gt_pos_i = matched_gt_positions[i]
            gt_quat_i = matched_gt_quaternions[i]  # [qx, qy, qz, qw]
            gt_rot_i = Rotation.from_quat(gt_quat_i).as_matrix()

            gt_pos_j = matched_gt_positions[j]
            gt_quat_j = matched_gt_quaternions[j]  # [qx, qy, qz, qw]
            gt_rot_j = Rotation.from_quat(gt_quat_j).as_matrix()

            gt_transform_i = np.eye(4)
            gt_transform_i[:3, :3] = gt_rot_i
            gt_transform_i[:3, 3] = gt_pos_i

            gt_transform_j = np.eye(4)
            gt_transform_j[:3, :3] = gt_rot_j
            gt_transform_j[:3, 3] = gt_pos_j

            gt_relative = np.linalg.inv(gt_transform_i) @ gt_transform_j

            # Estimated relative transformation
            est_pos_i = matched_est_positions[i]
            est_quat_i = matched_est_quaternions[i]  # [qx, qy, qz, qw]
            est_rot_i = Rotation.from_quat(est_quat_i).as_matrix()

            est_pos_j = matched_est_positions[j]
            est_quat_j = matched_est_quaternions[j]  # [qx, qy, qz, qw]
            est_rot_j = Rotation.from_quat(est_quat_j).as_matrix()

            est_transform_i = np.eye(4)
            est_transform_i[:3, :3] = est_rot_i
            est_transform_i[:3, 3] = est_pos_i

            est_transform_j = np.eye(4)
            est_transform_j[:3, :3] = est_rot_j
            est_transform_j[:3, 3] = est_pos_j

            est_relative = np.linalg.inv(est_transform_i) @ est_transform_j

            # Error
            error = np.linalg.inv(gt_relative) @ est_relative

            # Translation error
            translation_error = np.linalg.norm(error[:3, 3])
            translation_errors.append(translation_error)

            # Rotation error (in degrees)
            rotation_error_mat = error[:3, :3]
            rotation_error = np.arccos((np.trace(rotation_error_mat) - 1) / 2)
            rotation_error_deg = np.degrees(rotation_error)
            rotation_errors.append(rotation_error_deg)

        # Calculate statistics
        translation_errors = np.array(translation_errors)
        rotation_errors = np.array(rotation_errors)

        results = {
            "translation": {
                "rmse": np.sqrt(np.mean(np.square(translation_errors))),
                "mean": np.mean(translation_errors),
                "median": np.median(translation_errors),
                "std": np.std(translation_errors),
                "min": np.min(translation_errors),
                "max": np.max(translation_errors)
            },
            "rotation": {
                "rmse": np.sqrt(np.mean(np.square(rotation_errors))),
                "mean": np.mean(rotation_errors),
                "median": np.median(rotation_errors),
                "std": np.std(rotation_errors),
                "min": np.min(rotation_errors),
                "max": np.max(rotation_errors)
            },
            "num_pairs": len(translation_errors)
        }

        return results, translation_errors, rotation_errors

    def calculate_tracking_success_rate(self, algorithm_result_dir):
        """
        Calculate the tracking success rate based on the log files
        """
        log_file = None

        # Find the log file
        if (algorithm_result_dir / "stdout.txt").exists():
            log_file = algorithm_result_dir / "stdout.txt"
        elif (algorithm_result_dir / "log.txt").exists():
            log_file = algorithm_result_dir / "log.txt"

        if log_file is None:
            print(f"No log file found in {algorithm_result_dir}")
            return {
                "tracking_success_rate": 0.0,
                "tracking_failures": 0,
                "total_frames": 0
            }

        # Parse log file based on the algorithm
        if "orb_slam3" in str(algorithm_result_dir):
            return self._parse_orb_slam3_log(log_file)
        elif "dso" in str(algorithm_result_dir):
            return self._parse_dso_log(log_file)
        elif "svo" in str(algorithm_result_dir):
            return self._parse_svo_log(log_file)
        else:
            print(f"Unknown algorithm for log parsing: {algorithm_result_dir}")
            return {
                "tracking_success_rate": 0.0,
                "tracking_failures": 0,
                "total_frames": 0
            }

    def _parse_orb_slam3_log(self, log_file):
        total_frames = 0
        lost_frames = 0

        with open(log_file, 'r') as f:
            for line in f:
                if "Tracking frame" in line:
                    total_frames += 1
                if "TRACKING LOST" in line:
                    lost_frames += 1

        if total_frames == 0:
            return {
                "tracking_success_rate": 0.0,
                "tracking_failures": lost_frames,
                "total_frames": total_frames
            }

        tracking_success_rate = 1.0 - (lost_frames / total_frames)

        return {
            "tracking_success_rate": tracking_success_rate,
            "tracking_failures": lost_frames,
            "total_frames": total_frames
        }

    def _parse_dso_log(self, log_file):
        total_frames = 0
        lost_frames = 0

        with open(log_file, 'r') as f:
            for line in f:
                if "FRAME" in line:
                    total_frames += 1
                if "TRACKING LOST" in line:
                    lost_frames += 1

        if total_frames == 0:
            return {
                "tracking_success_rate": 0.0,
                "tracking_failures": lost_frames,
                "total_frames": total_frames
            }

        tracking_success_rate = 1.0 - (lost_frames / total_frames)

        return {
            "tracking_success_rate": tracking_success_rate,
            "tracking_failures": lost_frames,
            "total_frames": total_frames
        }

    def _parse_svo_log(self, log_file):
        total_frames = 0
        lost_frames = 0

        with open(log_file, 'r') as f:
            for line in f:
                if "Processing frame" in line:
                    total_frames += 1
                if "Lost" in line:
                    lost_frames += 1

        if total_frames == 0:
            return {
                "tracking_success_rate": 0.0,
                "tracking_failures": lost_frames,
                "total_frames": total_frames
            }

        tracking_success_rate = 1.0 - (lost_frames / total_frames)

        return {
            "tracking_success_rate": tracking_success_rate,
            "tracking_failures": lost_frames,
            "total_frames": total_frames
        }

    def evaluate_algorithm(self, algorithm_name, dataset_format, sensor_type=None):
        """
        Evaluate a SLAM algorithm on a dataset
        """
        # Construct the path to the ground truth and estimated trajectory
        if algorithm_name == "orb_slam3":
            if sensor_type is None:
                raise ValueError("sensor_type must be provided for ORB-SLAM3")

            result_dir = self.results_path / "orb_slam3" / dataset_format / sensor_type
            est_file = result_dir / "CameraTrajectory.txt"

        elif algorithm_name == "dso":
            result_dir = self.results_path / "dso" / dataset_format
            est_file = result_dir / "result.txt"

        elif algorithm_name == "svo":
            result_dir = self.results_path / "svo" / dataset_format
            est_file = result_dir / "trajectory.txt"

        else:
            raise ValueError(f"Unsupported algorithm: {algorithm_name}")

        # Check if estimated trajectory exists
        if not est_file.exists():
            print(f"Estimated trajectory file not found: {est_file}")
            return None

        # Get ground truth trajectory based on dataset format
        if dataset_format == "tum":
            gt_file = self.dataset_path / "preprocessed" / "tum_format" / "groundtruth.txt"
        elif dataset_format == "euroc":
            gt_file = self.dataset_path / "preprocessed" / "euroc_format" / "mav0" / "state_groundtruth_estimate0" / "data.csv"
            # Convert EuRoC ground truth to TUM format
            converted_gt_file = result_dir / "groundtruth.txt"
            self._convert_euroc_gt_to_tum(gt_file, converted_gt_file)
            gt_file = converted_gt_file
        elif dataset_format == "kitti":
            gt_file = self.dataset_path / "preprocessed" / "kitti_format" / "poses.txt"
            # Convert KITTI ground truth to TUM format
            converted_gt_file = result_dir / "groundtruth.txt"
            self._convert_kitti_gt_to_tum(gt_file, converted_gt_file)
            gt_file = converted_gt_file
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_format}")

        # Check if ground truth file exists
        if not gt_file.exists():
            print(f"Ground truth file not found: {gt_file}")
            return None

        # Calculate metrics
        ate_results, ate_errors = self.calculate_ate(gt_file, est_file)
        rpe_results, trans_errors, rot_errors = self.calculate_rpe(gt_file, est_file)
        tracking_results = self.calculate_tracking_success_rate(result_dir)

        # Combine results
        results = {
            "algorithm": algorithm_name,
            "dataset_format": dataset_format,
            "sensor_type": sensor_type,
            "ate": ate_results,
            "rpe": rpe_results,
            "tracking": tracking_results
        }

        # Save results to file
        output_file = result_dir / "evaluation_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)

        # Plot trajectory and errors
        self._plot_trajectory(gt_file, est_file, result_dir / "trajectory_plot.png")
        self._plot_ate_errors(ate_errors, result_dir / "ate_errors.png")
        self._plot_rpe_errors(trans_errors, rot_errors, result_dir / "rpe_errors.png")

        return results

    def _convert_euroc_gt_to_tum(self, euroc_file, tum_file):
        """
        Convert EuRoC ground truth to TUM format
        """
        data = pd.read_csv(euroc_file)

        with open(tum_file, 'w') as f:
            f.write("# timestamp tx ty tz qx qy qz qw\n")

            for _, row in data.iterrows():
                timestamp = row['timestamp'] / 1e9  # Convert from ns to s

                tx = row['p_RS_R_x']
                ty = row['p_RS_R_y']
                tz = row['p_RS_R_z']

                qx = row['q_RS_x']
                qy = row['q_RS_y']
                qz = row['q_RS_z']
                qw = row['q_RS_w']

                f.write(f"{timestamp:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")

    def _convert_kitti_gt_to_tum(self, kitti_file, tum_file):
        """
        Convert KITTI ground truth to TUM format
        """
        data = np.loadtxt(kitti_file)

        with open(tum_file, 'w') as f:
            f.write("# timestamp tx ty tz qx qy qz qw\n")

            for i, row in enumerate(data):
                # Use frame index as timestamp (KITTI doesn't provide timestamps)
                timestamp = float(i)

                # Extract rotation matrix and translation from the 3x4 matrix
                transform = row.reshape(3, 4)
                R = transform[:3, :3]
                t = transform[:3, 3]

                # Convert rotation matrix to quaternion
                quat = Rotation.from_matrix(R).as_quat()  # [x, y, z, w]

                tx, ty, tz = t
                qx, qy, qz, qw = quat

                f.write(f"{timestamp:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")

    def _plot_trajectory(self, gt_file, est_file, output_file):
        """
        Plot ground truth and estimated trajectories
        """
        # Load ground truth and estimated trajectories
        gt_data = np.loadtxt(gt_file)
        est_data = np.loadtxt(est_file)

        # Extract positions
        gt_positions = gt_data[:, 1:4]
        est_positions = est_data[:, 1:4]

        # Create figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot trajectories
        ax.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], 'r-', label='Ground Truth')
        ax.plot(est_positions[:, 0], est_positions[:, 1], est_positions[:, 2], 'b-', label='Estimated')

        # Set labels and title
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.set_title('Trajectory Comparison')
        ax.legend()

        # Save figure
        plt.savefig(output_file)
        plt.close(fig)

    def _plot_ate_errors(self, ate_errors, output_file):
        """
        Plot ATE errors
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot histogram
        ax.hist(ate_errors, bins=30, alpha=0.7)
        ax.axvline(np.mean(ate_errors), color='r', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(ate_errors):.3f}m')
        ax.axvline(np.median(ate_errors), color='g', linestyle='dashed', linewidth=2, label=f'Median: {np.median(ate_errors):.3f}m')

        # Set labels and title
        ax.set_xlabel('ATE [m]')
        ax.set_ylabel('Count')
        ax.set_title('Absolute Trajectory Error Distribution')
        ax.legend()

        # Save figure
        plt.savefig(output_file)
        plt.close(fig)

    def _plot_rpe_errors(self, trans_errors, rot_errors, output_file):
        """
        Plot RPE errors
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot translation errors
        ax1.hist(trans_errors, bins=30, alpha=0.7)
        ax1.axvline(np.mean(trans_errors), color='r', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(trans_errors):.3f}m')
        ax1.axvline(np.median(trans_errors), color='g', linestyle='dashed', linewidth=2, label=f'Median: {np.median(trans_errors):.3f}m')

        # Set labels and title
        ax1.set_xlabel('Translation Error [m]')
        ax1.set_ylabel('Count')
        ax1.set_title('Relative Translation Error Distribution')
        ax1.legend()

        # Plot rotation errors
        ax2.hist(rot_errors, bins=30, alpha=0.7)
        ax2.axvline(np.mean(rot_errors), color='r', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(rot_errors):.3f}°')
        ax2.axvline(np.median(rot_errors), color='g', linestyle='dashed', linewidth=2, label=f'Median: {np.median(rot_errors):.3f}°')

        # Set labels and title
        ax2.set_xlabel('Rotation Error [°]')
        ax2.set_ylabel('Count')
        ax2.set_title('Relative Rotation Error Distribution')
        ax2.legend()

        # Save figure
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close(fig)

    def compare_algorithms(self, results_list, output_dir=None):
        """
        Compare multiple algorithm results
        """
        if output_dir is None:
            output_dir = self.results_path / "comparison"

        output_dir.mkdir(exist_ok=True)

        # Extract metrics
        algorithms = []
        ate_rmse = []
        rpe_trans_rmse = []
        rpe_rot_rmse = []
        tracking_rates = []

        for results in results_list:
            algorithms.append(f"{results['algorithm']}-{results['sensor_type']}" if results['sensor_type'] else results['algorithm'])
            ate_rmse.append(results['ate']['rmse'])
            rpe_trans_rmse.append(results['rpe']['translation']['rmse'])
            rpe_rot_rmse.append(results['rpe']['rotation']['rmse'])
            tracking_rates.append(results['tracking']['tracking_success_rate'])

        # Create comparison plots
        self._plot_algorithm_comparison(
            algorithms,
            ate_rmse,
            "ATE RMSE [m]",
            output_dir / "ate_comparison.png"
        )

        self._plot_algorithm_comparison(
            algorithms,
            rpe_trans_rmse,
            "RPE Translation RMSE [m]",
            output_dir / "rpe_trans_comparison.png"
        )

        self._plot_algorithm_comparison(
            algorithms,
            rpe_rot_rmse,
            "RPE Rotation RMSE [°]",
            output_dir / "rpe_rot_comparison.png"
        )

        self._plot_algorithm_comparison(
            algorithms,
            tracking_rates,
            "Tracking Success Rate",
            output_dir / "tracking_comparison.png"
        )

        # Create summary table
        summary = {
            "Algorithm": algorithms,
            "ATE RMSE [m]": ate_rmse,
            "RPE Trans RMSE [m]": rpe_trans_rmse,
            "RPE Rot RMSE [°]": rpe_rot_rmse,
            "Tracking Rate": tracking_rates
        }

        df = pd.DataFrame(summary)

        # Save to CSV
        df.to_csv(output_dir / "summary.csv", index=False)

        # Create a simple HTML report
        html = """
        <html>
        <head>
            <title>SLAM Algorithm Comparison</title>
            <style>
                table { border-collapse: collapse; width: 100%; }
                th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #f2f2f2; }
                tr:hover { background-color: #f5f5f5; }
                img { max-width: 800px; display: block; margin: 20px auto; }
                .metric { margin-top: 40px; }
                h1, h2 { text-align: center; }
            </style>
        </head>
        <body>
            <h1>SLAM Algorithm Comparison</h1>

            <h2>Summary Table</h2>
            <table>
                <tr>
        """

        # Add table headers
        for col in df.columns:
            html += f"<th>{col}</th>"

        html += "</tr>"

        # Add table rows
        for _, row in df.iterrows():
            html += "<tr>"
            for col in df.columns:
                if col == "Algorithm":
                    html += f"<td>{row[col]}</td>"
                else:
                    html += f"<td>{row[col]:.4f}</td>"
            html += "</tr>"

        html += """
            </table>

            <div class="metric">
                <h2>Absolute Trajectory Error (ATE)</h2>
                <img src="ate_comparison.png" alt="ATE Comparison">
            </div>

            <div class="metric">
                <h2>Relative Pose Error - Translation</h2>
                <img src="rpe_trans_comparison.png" alt="RPE Translation Comparison">
            </div>

            <div class="metric">
                <h2>Relative Pose Error - Rotation</h2>
                <img src="rpe_rot_comparison.png" alt="RPE Rotation Comparison">
            </div>

            <div class="metric">
                <h2>Tracking Success Rate</h2>
                <img src="tracking_comparison.png" alt="Tracking Rate Comparison">
            </div>
        </body>
        </html>
        """

        # Save HTML report
        with open(output_dir / "report.html", 'w') as f:
            f.write(html)

        return output_dir

    def _plot_algorithm_comparison(self, algorithms, values, ylabel, output_file):
        """
        Plot algorithm comparison bar chart
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create bar chart
        bars = ax.bar(algorithms, values, alpha=0.7, color='skyblue')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{height:.4f}',
                ha='center',
                va='bottom'
            )

        # Set labels and title
        ax.set_xlabel('Algorithm')
        ax.set_ylabel(ylabel)
        ax.set_title(f'Algorithm Comparison - {ylabel}')

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')

        # Adjust layout
        plt.tight_layout()

        # Save figure
        plt.savefig(output_file)
        plt.close(fig)

    def run_and_evaluate_all(self, dataset_formats=["tum"], sensor_types=["mono"]):
        """
        Run and evaluate all configured algorithms
        """
        results_list = []

        for dataset_format in dataset_formats:
            # Run and evaluate ORB-SLAM3
            for sensor_type in sensor_types:
                try:
                    orb_result_dir = self.run_orb_slam3(dataset_format, sensor_type)
                    if orb_result_dir:
                        orb_results = self.evaluate_algorithm("orb_slam3", dataset_format, sensor_type)
                        if orb_results:
                            results_list.append(orb_results)
                except Exception as e:
                    print(f"Error with ORB-SLAM3 ({dataset_format}, {sensor_type}): {e}")

            # Run and evaluate DSO
            try:
                dso_result_dir = self.run_dso(dataset_format)
                if dso_result_dir:
                    dso_results = self.evaluate_algorithm("dso", dataset_format)
                    if dso_results:
                        results_list.append(dso_results)
            except Exception as e:
                print(f"Error with DSO ({dataset_format}): {e}")

            # Run and evaluate SVO
            try:
                svo_result_dir = self.run_svo(dataset_format)
                if svo_result_dir:
                    svo_results = self.evaluate_algorithm("svo", dataset_format)
                    if svo_results:
                        results_list.append(svo_results)
            except Exception as e:
                print(f"Error with SVO ({dataset_format}): {e}")

        # Compare algorithms
        if results_list:
            comparison_dir = self.compare_algorithms(results_list)
            print(f"Comparison results saved to: {comparison_dir}")

        return results_list

# Example usage
if __name__ == "__main__":
    dataset_path = "./data/urban_canyon_dataset"

    evaluator = SLAMEvaluator(dataset_path)
    results = evaluator.run_and_evaluate_all(dataset_formats=["tum"], sensor_types=["mono"])
    print(f"Evaluation completed with {len(results)} algorithm results.")
