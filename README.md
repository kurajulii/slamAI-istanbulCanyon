# Enhancing Drone Visual Odometry/SLAM Robustness in Simulated İstanbul Urban Canyons

This project aims to improve the robustness of Visual Odometry (VO) and Visual SLAM (Simultaneous Localization and Mapping) algorithms for drone navigation in challenging urban canyon environments, with a particular focus on İstanbul-like settings. The project uses machine learning techniques to enhance feature detection, description, and loop closure under difficult conditions like poor lighting, textureless surfaces, and GPS-denied environments.

## Project Structure

The codebase is organized into the following modules:

-  **airsim_setup.py**: Sets up the AirSim simulation environment for creating İstanbul-like urban canyons.
-  **data_preprocessing.py**: Processes data collected from the simulation environment into formats suitable for SLAM algorithms and ML training.
-  **baseline_slam.py**: Implements and evaluates baseline SLAM algorithms (ORB-SLAM3, DSO, SVO).
-  **ml_enhancement.py**: Machine learning models for enhancing feature detection, loop closure, and patch description.
-  **slam_integration.py**: Integrates ML models with ORB-SLAM3 for enhanced performance.
-  **main.py**: Main script to orchestrate the entire workflow.

## Installation Requirements

### Dependencies

```
numpy
opencv-python
torch
matplotlib
airsim
scipy
pillow
onnxruntime
scikit-learn
pandas
```

### External SLAM Libraries

The project requires the following external SLAM libraries to be installed:

1. **ORB-SLAM3**: Clone from https://github.com/UZ-SLAMLab/ORB_SLAM3 and follow installation instructions.
2. **DSO (Direct Sparse Odometry)** (optional): Clone from https://github.com/JakobEngel/dso and follow installation instructions.
3. **SVO (Semi-direct Visual Odometry)** (optional): Clone from https://github.com/uzh-rpg/rpg_svo and follow installation instructions.

### AirSim

This project uses AirSim for drone simulation. Follow the installation instructions at https://microsoft.github.io/AirSim/

## Usage

### Full Pipeline

To run the entire pipeline (simulation, preprocessing, ML training, evaluation):

```bash
python main.py --mode full_pipeline --output_dir ./output --orb_slam_path ./ORB_SLAM3 --dataset_format tum --sensor_type mono --scenario urban_canyon
```

### Individual Steps

1. **Run Simulation Only**:

```bash
python main.py --mode simulate --output_dir ./output --scenario urban_canyon
```

2. **Preprocess Dataset**:

```bash
python main.py --mode preprocess --output_dir ./output --dataset_format tum
```

3. **Train ML Models**:

```bash
python main.py --mode train_ml --output_dir ./output --ml_batch_size 16 --ml_epochs 50 --gpu_id 0
```

4. **Evaluate Baseline SLAM**:

```bash
python main.py --mode evaluate_baseline --output_dir ./output --orb_slam_path ./ORB_SLAM3 --dataset_format tum --sensor_type mono
```

5. **Run Enhanced SLAM**:

```bash
python main.py --mode enhance_slam --output_dir ./output --orb_slam_path ./ORB_SLAM3 --dataset_format tum --sensor_type mono
```

6. **Evaluate Enhanced SLAM**:

```bash
python main.py --mode evaluate_enhanced --output_dir ./output --dataset_format tum
```

## ML Enhancement Approaches

The project implements three main ML approaches to improve SLAM robustness:

1. **Feature Enhancement Network**: A CNN-based image enhancement network that improves feature visibility in challenging lighting conditions.

2. **Deep Loop Closure Detection**: A deep learning network that generates robust embeddings for place recognition, improving loop closure in similar-looking urban environments.

3. **Robust Patch Descriptor Network**: A network that generates descriptors for image patches that are more robust to lighting changes and viewpoint variations.

## Results and Evaluation

The system evaluates both baseline and ML-enhanced SLAM algorithms using standard metrics:

-  **Absolute Trajectory Error (ATE)**: Measures the absolute difference between estimated and ground truth poses.
-  **Relative Pose Error (RPE)**: Measures the relative difference between pose pairs.
-  **Tracking Success Rate**: Percentage of frames where tracking was successfully maintained.

Results are presented in both numerical form and visualizations, making it easy to compare the performance of baseline and enhanced algorithms.

## Simulated Environments

The project includes code to simulate various challenging scenarios:

1. **Urban Canyon**: Narrow streets with tall buildings causing GPS shadowing.
2. **Dynamic Lighting**: Transitions between bright sunlight and deep shadows.
3. **Textureless**: Large surfaces with minimal texture for feature detection.

## Contributing

Contributions to this project are welcome. Please ensure that any pull requests maintain the coding style and include appropriate tests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

-  The ORB-SLAM3 authors for their excellent SLAM system.
-  Microsoft for the AirSim simulator.
-  The PyTorch team for their deep learning framework.
