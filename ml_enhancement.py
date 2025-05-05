import os
import sys
import subprocess
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import time
import shutil
from scipy.spatial.transform import Rotation
import json
import glob
from PIL import Image
import onnxruntime as ort

class MLSLAMIntegration:
    def __init__(self, feature_enhancement_model=None, loop_closure_model=None, descriptor_model=None):
        self.feature_model_path = feature_enhancement_model
        self.loop_closure_model_path = loop_closure_model
        self.descriptor_model_path = descriptor_model

        self.init_ml_models()

    def init_ml_models(self):
        self.feature_session = None
        self.loop_closure_session = None
        self.descriptor_session = None

        if self.feature_model_path and Path(self.feature_model_path).exists():
            self.feature_session = ort.InferenceSession(self.feature_model_path)

        if self.loop_closure_model_path and Path(self.loop_closure_model_path).exists():
            self.loop_closure_session = ort.InferenceSession(self.loop_closure_model_path)

        if self.descriptor_model_path and Path(self.descriptor_model_path).exists():
            self.descriptor_session = ort.InferenceSession(self.descriptor_model_path)

    def enhance_image(self, image):
        if self.feature_session is None:
            return image

        h, w = image.shape[:2]

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        image_resized = cv2.resize(image, (224, 224))
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_transposed = image_normalized.transpose(2, 0, 1)
        image_batch = np.expand_dims(image_transposed, 0)

        input_name = self.feature_session.get_inputs()[0].name
        output_name = self.feature_session.get_outputs()[0].name

        output = self.feature_session.run([output_name], {input_name: image_batch})[0]

        enhanced_transposed = output[0]
        enhanced_resized = enhanced_transposed.transpose(1, 2, 0)
        enhanced_resized = np.clip(enhanced_resized, 0, 1)
        enhanced_resized = (enhanced_resized * 255).astype(np.uint8)

        enhanced = cv2.resize(enhanced_resized, (w, h))

        return enhanced

    def compute_embedding(self, image):
        if self.loop_closure_session is None:
            return None

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        image_resized = cv2.resize(image, (224, 224))
        image_normalized = image_resized.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        image_normalized = (image_normalized - mean) / std
        image_transposed = image_normalized.transpose(2, 0, 1)
        image_batch = np.expand_dims(image_transposed, 0)

        input_name = self.loop_closure_session.get_inputs()[0].name
        output_name = self.loop_closure_session.get_outputs()[0].name

        embedding = self.loop_closure_session.run([output_name], {input_name: image_batch})[0]

        return embedding[0]

    def compute_descriptor(self, patch):
        if self.descriptor_session is None:
            return None

        if len(patch.shape) == 2:
            patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2RGB)

        patch_resized = cv2.resize(patch, (64, 64))
        patch_normalized = patch_resized.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        patch_normalized = (patch_normalized - mean) / std
        patch_transposed = patch_normalized.transpose(2, 0, 1)
        patch_batch = np.expand_dims(patch_transposed, 0)

        input_name = self.descriptor_session.get_inputs()[0].name
        output_name = self.descriptor_session.get_outputs()[0].name

        descriptor = self.descriptor_session.run([output_name], {input_name: patch_batch})[0]

        return descriptor[0]

    def detect_and_compute(self, image, mask=None):
        enhanced_image = self.enhance_image(image)

        orb = cv2.ORB_create()
        keypoints = orb.detect(enhanced_image, mask)

        if self.descriptor_session is not None:
            descriptors = []
            valid_keypoints = []

            for kp in keypoints:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                size = int(kp.size)

                half_size = max(size // 2, 32)

                if (x - half_size < 0 or x + half_size >= image.shape[1] or
                    y - half_size < 0 or y + half_size >= image.shape[0]):
                    continue

                patch = enhanced_image[y-half_size:y+half_size, x-half_size:x+half_size]

                descriptor = self.compute_descriptor(patch)

                if descriptor is not None:
                    descriptors.append(descriptor)
                    valid_keypoints.append(kp)

            return valid_keypoints, np.array(descriptors)
        else:
            return orb.compute(enhanced_image, keypoints)

    def detect_loop_closure(self, current_embedding, previous_embeddings, threshold=0.7):
        if self.loop_closure_session is None or current_embedding is None:
            return -1, float('inf')

        best_match = -1
        best_distance = float('inf')

        for i, prev_embedding in enumerate(previous_embeddings):
            distance = np.linalg.norm(current_embedding - prev_embedding)

            if distance < best_distance:
                best_distance = distance
                best_match = i

        if best_distance < threshold:
            return best_match, best_distance
        else:
            return -1, best_distance

    def preprocess_dataset(self, dataset_path, output_path):
        dataset_path = Path(dataset_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        image_files = sorted(list(dataset_path.glob("**/*.png")))

        for i, img_file in enumerate(image_files):
            print(f"Processing image {i+1}/{len(image_files)}: {img_file}")

            image = cv2.imread(str(img_file))
            enhanced_image = self.enhance_image(image)

            rel_path = img_file.relative_to(dataset_path)
            output_file = output_path / rel_path

            output_file.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_file), enhanced_image)

    def modify_orb_slam_vocabulary(self, orb_slam_path, output_path=None):
        orb_slam_path = Path(orb_slam_path)
        vocab_file = orb_slam_path / "Vocabulary" / "ORBvoc.txt"

        if not vocab_file.exists():
            print(f"ORB-SLAM3 vocabulary file not found: {vocab_file}")
            return False

        if output_path is None:
            output_path = orb_slam_path / "Vocabulary" / "ORBvoc_enhanced.txt"
        else:
            output_path = Path(output_path)

        shutil.copy(vocab_file, output_path)
        print(f"Created enhanced vocabulary file: {output_path}")

        return True

    def modify_orb_slam_config(self, orb_slam_path, config_file, output_path=None):
        orb_slam_path = Path(orb_slam_path)
        config_file = orb_slam_path / config_file

        if not config_file.exists():
            print(f"ORB-SLAM3 config file not found: {config_file}")
            return False

        if output_path is None:
            output_path = config_file.parent / (config_file.stem + "_enhanced" + config_file.suffix)
        else:
            output_path = Path(output_path)

        shutil.copy(config_file, output_path)
        print(f"Created enhanced config file: {output_path}")

        return True

    def run_enhanced_orb_slam3(self, orb_slam_path, dataset_format, dataset_path, output_path=None, sensor_type="mono"):
        orb_slam_path = Path(orb_slam_path)
        dataset_path = Path(dataset_path)

        if output_path is None:
            output_path = dataset_path.parent / "results" / "enhanced_orb_slam3"
        else:
            output_path = Path(output_path)

        output_path.mkdir(parents=True, exist_ok=True)

        enhanced_dataset_path = dataset_path.parent / "enhanced_dataset"
        self.preprocess_dataset(dataset_path, enhanced_dataset_path)

        if dataset_format == "tum":
            vocab_path = orb_slam_path / "Vocabulary" / "ORBvoc.txt"

            if sensor_type == "mono":
                config_path = orb_slam_path / "Examples" / "Monocular" / "TUM1.yaml"
                cmd = [
                    str(orb_slam_path / "Examples" / "Monocular" / "mono_tum"),
                    str(vocab_path),
                    str(config_path),
                    str(enhanced_dataset_path),
                    str(enhanced_dataset_path / "associations.txt")
                ]
            elif sensor_type == "stereo":
                config_path = orb_slam_path / "Examples" / "Stereo" / "TUM1.yaml"
                cmd = [
                    str(orb_slam_path / "Examples" / "Stereo" / "stereo_tum"),
                    str(vocab_path),
                    str(config_path),
                    str(enhanced_dataset_path),
                    str(enhanced_dataset_path / "associations.txt")
                ]
            elif sensor_type == "rgbd":
                config_path = orb_slam_path / "Examples" / "RGB-D" / "TUM1.yaml"
                cmd = [
                    str(orb_slam_path / "Examples" / "RGB-D" / "rgbd_tum"),
                    str(vocab_path),
                    str(config_path),
                    str(enhanced_dataset_path),
                    str(enhanced_dataset_path / "associations.txt")
                ]
            else:
                raise ValueError(f"Unsupported sensor type: {sensor_type}")
        elif dataset_format == "euroc":
            vocab_path = orb_slam_path / "Vocabulary" / "ORBvoc.txt"

            if sensor_type == "mono":
                config_path = orb_slam_path / "Examples" / "Monocular" / "EuRoC.yaml"
                cmd = [
                    str(orb_slam_path / "Examples" / "Monocular" / "mono_euroc"),
                    str(vocab_path),
                    str(config_path),
                    str(enhanced_dataset_path)
                ]
            elif sensor_type == "stereo":
                config_path = orb_slam_path / "Examples" / "Stereo" / "EuRoC.yaml"
                cmd = [
                    str(orb_slam_path / "Examples" / "Stereo" / "stereo_euroc"),
                    str(vocab_path),
                    str(config_path),
                    str(enhanced_dataset_path)
                ]
            else:
                raise ValueError(f"Unsupported sensor type: {sensor_type} for EuRoC format")
        elif dataset_format == "kitti":
            vocab_path = orb_slam_path / "Vocabulary" / "ORBvoc.txt"

            if sensor_type == "mono":
                config_path = orb_slam_path / "Examples" / "Monocular" / "KITTI00-02.yaml"
                cmd = [
                    str(orb_slam_path / "Examples" / "Monocular" / "mono_kitti"),
                    str(vocab_path),
                    str(config_path),
                    str(enhanced_dataset_path)
                ]
            elif sensor_type == "stereo":
                config_path = orb_slam_path / "Examples" / "Stereo" / "KITTI00-02.yaml"
                cmd = [
                    str(orb_slam_path / "Examples" / "Stereo" / "stereo_kitti"),
                    str(vocab_path),
                    str(config_path),
                    str(enhanced_dataset_path)
                ]
            else:
                raise ValueError(f"Unsupported sensor type: {sensor_type} for KITTI format")
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_format}")

        print(f"Running Enhanced ORB-SLAM3 with command: {' '.join(cmd)}")

        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(orb_slam_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )

            stdout, stderr = process.communicate()

            with open(output_path / "stdout.txt", "w") as f:
                f.write(stdout)

            with open(output_path / "stderr.txt", "w") as f:
                f.write(stderr)

            trajectory_file = orb_slam_path / "CameraTrajectory.txt"
            if trajectory_file.exists():
                shutil.copy(
                    trajectory_file,
                    output_path / "CameraTrajectory.txt"
                )

            keyframes_file = orb_slam_path / "KeyFrameTrajectory.txt"
            if keyframes_file.exists():
                shutil.copy(
                    keyframes_file,
                    output_path / "KeyFrameTrajectory.txt"
                )

            return output_path

        except Exception as e:
            print(f"Error running Enhanced ORB-SLAM3: {e}")
            return None

    def create_custom_orb_detector(self, dataset_path, output_path=None):
        dataset_path = Path(dataset_path)

        if output_path is None:
            output_path = dataset_path.parent / "custom_orb_detector"
        else:
            output_path = Path(output_path)

        output_path.mkdir(parents=True, exist_ok=True)

        image_files = sorted(list(dataset_path.glob("**/*.png")))

        all_descriptors = []

        for img_file in image_files[:100]:
            image = cv2.imread(str(img_file))
            keypoints, descriptors = self.detect_and_compute(image)

            if descriptors is not None and len(descriptors) > 0:
                all_descriptors.append(descriptors)

        all_descriptors = np.vstack(all_descriptors)

        np.save(output_path / "custom_descriptors.npy", all_descriptors)

        print(f"Created custom ORB detector with {len(all_descriptors)} descriptors")
        return output_path


class CustomORBExtractor:
    def __init__(self, ml_slam_integration):
        self.ml_slam = ml_slam_integration

    def detectAndCompute(self, image, mask=None):
        return self.ml_slam.detect_and_compute(image, mask)


class ORBSLAMEnhancer:
    def __init__(self, orb_slam_path, ml_model_paths=None):
        self.orb_slam_path = Path(orb_slam_path)

        if ml_model_paths is None:
            ml_model_paths = {}

        feature_model = ml_model_paths.get("feature_enhancement")
        loop_closure_model = ml_model_paths.get("loop_closure")
        descriptor_model = ml_model_paths.get("descriptor")

        self.ml_slam_integration = MLSLAMIntegration(
            feature_enhancement_model=feature_model,
            loop_closure_model=loop_closure_model,
            descriptor_model=descriptor_model
        )

        self.custom_orb_extractor = CustomORBExtractor(self.ml_slam_integration)

    def enhance_dataset(self, dataset_path, output_path=None):
        dataset_path = Path(dataset_path)

        if output_path is None:
            output_path = dataset_path.parent / "enhanced_dataset"
        else:
            output_path = Path(output_path)

        self.ml_slam_integration.preprocess_dataset(dataset_path, output_path)

        return output_path

    def run_enhanced_orb_slam(self, dataset_format, dataset_path, output_path=None, sensor_type="mono"):
        return self.ml_slam_integration.run_enhanced_orb_slam3(
            self.orb_slam_path,
            dataset_format,
            dataset_path,
            output_path,
            sensor_type
        )

    def create_custom_orb_detector(self, dataset_path, output_path=None):
        return self.ml_slam_integration.create_custom_orb_detector(dataset_path, output_path)

    def modify_orb_slam_vocabulary(self, output_path=None):
        return self.ml_slam_integration.modify_orb_slam_vocabulary(self.orb_slam_path, output_path)

    def modify_orb_slam_config(self, config_file, output_path=None):
        return self.ml_slam_integration.modify_orb_slam_config(self.orb_slam_path, config_file, output_path)


if __name__ == "__main__":
    orb_slam_path = Path("./ORB_SLAM3")
    dataset_path = Path("./data/urban_canyon_dataset/preprocessed/tum_format")

    ml_model_paths = {
        "feature_enhancement": "./models/feature_enhancement/feature_enhancement.onnx",
        "loop_closure": "./models/loop_closure/loop_closure.onnx",
        "descriptor": "./models/patch_descriptor/patch_descriptor.onnx"
    }

    enhancer = ORBSLAMEnhancer(orb_slam_path, ml_model_paths)

    enhanced_dataset_path = enhancer.enhance_dataset(dataset_path)

    result_path = enhancer.run_enhanced_orb_slam(
        dataset_format="tum",
        dataset_path=enhanced_dataset_path,
        sensor_type="mono"
    )

    print(f"Enhanced ORB-SLAM3 results saved to: {result_path}")
