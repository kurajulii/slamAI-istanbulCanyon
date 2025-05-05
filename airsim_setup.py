import os
import sys
import time
import numpy as np
import cv2
import airsim
import math
from pathlib import Path

class IstanbulUrbanCanyonsSim:
    def __init__(self, settings_path=None):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self.base_path = Path("./data")
        self.base_path.mkdir(exist_ok=True)

        self.trajectory_data = []
        self.configure_simulation(settings_path)

    def configure_simulation(self, settings_path=None):
        if settings_path:
            self.client.simLoadLevel(settings_path)
            time.sleep(2)

        camera_settings = airsim.ImageRequest(
            0, airsim.ImageType.Scene, False, False
        )
        depth_settings = airsim.ImageRequest(
            0, airsim.ImageType.DepthVis, False, False
        )

        self.image_requests = [camera_settings, depth_settings]

        imu_data = self.client.getImuData()
        self.imu_frequency = 200  # Hz

    def reset_environment(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        time.sleep(1)
        self.trajectory_data = []

    def takeoff(self, min_altitude=3.0):
        self.client.takeoffAsync().join()

    def move_to_position(self, x, y, z, velocity=5):
        self.client.moveToPositionAsync(x, y, z, velocity).join()

    def follow_trajectory(self, trajectory, velocity=5):
        for waypoint in trajectory:
            self.client.moveToPositionAsync(
                waypoint[0], waypoint[1], waypoint[2], velocity
            ).join()

            pose = self.client.simGetVehiclePose()
            gt_position = [pose.position.x_val, pose.position.y_val, pose.position.z_val]
            gt_orientation = [
                pose.orientation.w_val,
                pose.orientation.x_val,
                pose.orientation.y_val,
                pose.orientation.z_val
            ]

            self.trajectory_data.append({
                'timestamp': time.time(),
                'gt_position': gt_position,
                'gt_orientation': gt_orientation
            })

    def generate_exploration_trajectory(self, radius=50, height=-20, points=10):
        trajectory = []
        for i in range(points):
            angle = i * 2 * math.pi / points
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            z = height
            trajectory.append([x, y, z])

        return trajectory

    def generate_urban_canyon_trajectory(self, start_pos, end_pos, altitude=-20, points=20):
        trajectory = []
        for i in range(points):
            t = i / (points - 1)
            x = start_pos[0] * (1 - t) + end_pos[0] * t
            y = start_pos[1] * (1 - t) + end_pos[1] * t

            # Add slight variation in altitude to simulate realistic flight
            z_variation = np.random.normal(0, 0.5)
            z = altitude + z_variation

            trajectory.append([x, y, z])

        return trajectory

    def collect_sensor_data(self, output_folder=None):
        if output_folder:
            data_path = self.base_path / output_folder
            data_path.mkdir(exist_ok=True)
        else:
            data_path = self.base_path / f"flight_data_{int(time.time())}"
            data_path.mkdir(exist_ok=True)

        image_path = data_path / "images"
        image_path.mkdir(exist_ok=True)

        depth_path = data_path / "depth"
        depth_path.mkdir(exist_ok=True)

        metadata_file = data_path / "trajectory.npy"

        frame_count = 0

        try:
            while True:
                # Get drone pose
                pose = self.client.simGetVehiclePose()
                gt_position = [pose.position.x_val, pose.position.y_val, pose.position.z_val]
                gt_orientation = [
                    pose.orientation.w_val,
                    pose.orientation.x_val,
                    pose.orientation.y_val,
                    pose.orientation.z_val
                ]

                # Get IMU data
                imu_data = self.client.getImuData()
                angular_velocity = [
                    imu_data.angular_velocity.x_val,
                    imu_data.angular_velocity.y_val,
                    imu_data.angular_velocity.z_val
                ]
                linear_acceleration = [
                    imu_data.linear_acceleration.x_val,
                    imu_data.linear_acceleration.y_val,
                    imu_data.linear_acceleration.z_val
                ]

                # Get camera images
                responses = self.client.simGetImages(self.image_requests)

                # Process RGB image
                if responses[0].pixels_as_float:
                    rgb_image = np.array(responses[0].image_data_float, dtype=np.float32)
                else:
                    rgb_image = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                    rgb_image = rgb_image.reshape(responses[0].height, responses[0].width, 3)

                # Process depth image
                if responses[1].pixels_as_float:
                    depth_image = np.array(responses[1].image_data_float, dtype=np.float32)
                    depth_image = depth_image.reshape(responses[1].height, responses[1].width)
                else:
                    depth_image = np.frombuffer(responses[1].image_data_uint8, dtype=np.uint8)
                    depth_image = depth_image.reshape(responses[1].height, responses[1].width, 3)

                # Save data
                timestamp = time.time()
                cv2.imwrite(str(image_path / f"{frame_count:06d}.png"), rgb_image)
                np.save(str(depth_path / f"{frame_count:06d}.npy"), depth_image)

                self.trajectory_data.append({
                    'timestamp': timestamp,
                    'frame_id': frame_count,
                    'gt_position': gt_position,
                    'gt_orientation': gt_orientation,
                    'angular_velocity': angular_velocity,
                    'linear_acceleration': linear_acceleration
                })

                frame_count += 1
                time.sleep(0.05)  # 20Hz data collection

        except KeyboardInterrupt:
            print(f"Data collection stopped. Collected {frame_count} frames.")

        # Save trajectory data
        np.save(str(metadata_file), self.trajectory_data)

        # Create a basic calibration file
        camera_info = {
            'width': responses[0].width,
            'height': responses[0].height,
            'fov': 90.0,  # Default FOV in AirSim
            'fx': responses[0].width / 2 / math.tan(math.radians(90.0 / 2)),
            'fy': responses[0].height / 2 / math.tan(math.radians(90.0 / 2)),
            'cx': responses[0].width / 2,
            'cy': responses[0].height / 2
        }

        np.save(str(data_path / "camera_info.npy"), camera_info)

        return data_path

    def generate_scenario(self, scenario_type="urban_canyon"):
        if scenario_type == "urban_canyon":
            # Simulate flying through a narrow street
            start_pos = [0, 0]
            end_pos = [100, 0]
            trajectory = self.generate_urban_canyon_trajectory(start_pos, end_pos)

            # Add obstacles
            # This would be done through the Unreal Engine editor by placing buildings

            return trajectory

        elif scenario_type == "dynamic_lighting":
            # Simulate flying from shadow to bright light
            trajectory = self.generate_exploration_trajectory()

            # Lighting changes would be part of the Unreal environment setup

            return trajectory

        elif scenario_type == "textureless":
            # Simulate flying by large, textureless walls
            start_pos = [0, 0]
            end_pos = [50, 50]
            trajectory = self.generate_urban_canyon_trajectory(start_pos, end_pos)

            # Textureless surfaces would be part of the Unreal environment

            return trajectory

        else:
            raise ValueError(f"Unknown scenario type: {scenario_type}")

# Example usage
if __name__ == "__main__":
    sim = IstanbulUrbanCanyonsSim()
    sim.reset_environment()
    sim.takeoff()

    scenario = "urban_canyon"
    trajectory = sim.generate_scenario(scenario)

    print(f"Flying trajectory for scenario: {scenario}")
    sim.follow_trajectory(trajectory)

    print("Collecting sensor data...")
    data_path = sim.collect_sensor_data(f"{scenario}_dataset")

    print(f"Dataset saved to: {data_path}")

    sim.client.armDisarm(False)
    sim.client.enableApiControl(False)
