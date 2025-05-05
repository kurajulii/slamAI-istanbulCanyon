import os
import sys
import argparse
import time
import json
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil

from airsim_setup import IstanbulUrbanCanyonsSim
from data_preprocessing import SLAMDataPreprocessor
from baseline_slam import SLAMEvaluator
from ml_enhancement import FeatureEnhancementTrainer, DeepLoopClosureTrainer, RobustPatchDescriptorTrainer
from slam_integration import ORBSLAMEnhancer

def parse_args():
    parser = argparse.ArgumentParser(description="Drone Visual SLAM Enhancement Pipeline")

    parser.add_argument("--mode", type=str, required=True, choices=[
        "simulate", "preprocess", "train_ml", "evaluate_baseline", "enhance_slam", "evaluate_enhanced", "full_pipeline"
    ], help="Operation mode")

    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--orb_slam_path", type=str, default="./ORB_SLAM3", help="Path to ORB-SLAM3")
    parser.add_argument("--dataset_format", type=str, default="tum", choices=["tum", "euroc", "kitti"], help="Dataset format")
    parser.add_argument("--sensor_type", type=str, default="mono", choices=["mono", "stereo", "rgbd"], help="Sensor type")
    parser.add_argument("--scenario", type=str, default="urban_canyon", help="Simulation scenario type")
    parser.add_argument("--ml_batch_size", type=int, default=16, help="ML training batch size")
    parser.add_argument("--ml_epochs", type=int, default=50, help="ML training epochs")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID (-1 for CPU)")

    return parser.parse_args()

def setup_environment(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True)

    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)

    results_dir = output_dir / "results"
    results_dir.mkdir(exist_ok=True)

    logs_dir = output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    return {
        "output_dir": output_dir,
        "data_dir": data_dir,
        "models_dir": models_dir,
        "results_dir": results_dir,
        "logs_dir": logs_dir
    }

def run_simulation(args, dirs):
    scenario = args.scenario
    dataset_name = f"{scenario}_dataset"

    sim = IstanbulUrbanCanyonsSim()
    sim.reset_environment()
    sim.takeoff()

    trajectory = sim.generate_scenario(scenario)

    print(f"Flying trajectory for scenario: {scenario}")
    sim.follow_trajectory(trajectory)

    print("Collecting sensor data...")
    data_path = sim.collect_sensor_data(dataset_name)

    dataset_target = dirs["data_dir"] / dataset_name

    if data_path != dataset_target:
        if dataset_target.exists():
            shutil.rmtree(dataset_target)

        shutil.copytree(data_path, dataset_target)

    sim.client.armDisarm(False)
    sim.client.enableApiControl(False)

    return dataset_target

def preprocess_data(args, dirs, dataset_path=None):
    if dataset_path is None:
        scenario = args.scenario
        dataset_name = f"{scenario}_dataset"
        dataset_path = dirs["data_dir"] / dataset_name

    preprocessor = SLAMDataPreprocessor(dataset_path)

    print("Preprocessing dataset...")
    output_paths = preprocessor.process_dataset(
        formats=[args.dataset_format],
        create_ml_dataset=True
    )

    return output_paths

def train_ml_models(args, dirs, ml_dataset_path=None):
    if ml_dataset_path is None:
        scenario = args.scenario
        dataset_name = f"{scenario}_dataset"
        dataset_path = dirs["data_dir"] / dataset_name
        ml_dataset_path = dataset_path / "preprocessed" / "ml_dataset"

    feature_model_dir = dirs["models_dir"] / "feature_enhancement"
    feature_model_dir.mkdir(exist_ok=True)

    loop_closure_model_dir = dirs["models_dir"] / "loop_closure"
    loop_closure_model_dir.mkdir(exist_ok=True)

    patch_model_dir = dirs["models_dir"] / "patch_descriptor"
    patch_model_dir.mkdir(exist_ok=True)

    print("Training Feature Enhancement Network...")
    feature_trainer = FeatureEnhancementTrainer(
        dataset_path=ml_dataset_path,
        batch_size=args.ml_batch_size,
        learning_rate=0.001,
        num_epochs=args.ml_epochs
    )

    feature_trainer.train()
    feature_trainer.save_model("best_model.pth")
    feature_trainer.export_model("feature_enhancement.onnx")

    for file in feature_model_dir.parent.glob("feature_enhancement/*.pth"):
        shutil.copy(file, feature_model_dir)

    for file in feature_model_dir.parent.glob("feature_enhancement/*.onnx"):
        shutil.copy(file, feature_model_dir)

    print("Training Loop Closure Network...")
    loop_closure_trainer = DeepLoopClosureTrainer(
        dataset_path=ml_dataset_path,
        batch_size=args.ml_batch_size,
        learning_rate=0.001,
        num_epochs=args.ml_epochs,
        embedding_dim=256
    )

    loop_closure_trainer.train()
    loop_closure_trainer.save_model("best_model.pth")
    loop_closure_trainer.export_model("loop_closure.onnx")

    for file in loop_closure_model_dir.parent.glob("loop_closure/*.pth"):
        shutil.copy(file, loop_closure_model_dir)

    for file in loop_closure_model_dir.parent.glob("loop_closure/*.onnx"):
        shutil.copy(file, loop_closure_model_dir)

    print("Training Robust Patch Descriptor Network...")
    patch_trainer = RobustPatchDescriptorTrainer(
        dataset_path=ml_dataset_path,
        batch_size=args.ml_batch_size,
        learning_rate=0.001,
        num_epochs=args.ml_epochs,
        descriptor_dim=128
    )

    patch_trainer.train()
    patch_trainer.save_model("best_model.pth")
    patch_trainer.export_model("patch_descriptor.onnx")

    for file in patch_model_dir.parent.glob("patch_descriptor/*.pth"):
        shutil.copy(file, patch_model_dir)

    for file in patch_model_dir.parent.glob("patch_descriptor/*.onnx"):
        shutil.copy(file, patch_model_dir)

    model_paths = {
        "feature_enhancement": str(feature_model_dir / "feature_enhancement.onnx"),
        "loop_closure": str(loop_closure_model_dir / "loop_closure.onnx"),
        "descriptor": str(patch_model_dir / "patch_descriptor.onnx")
    }

    return model_paths

def evaluate_baseline_slam(args, dirs, dataset_path=None):
    if dataset_path is None:
        scenario = args.scenario
        dataset_name = f"{scenario}_dataset"
        dataset_path = dirs["data_dir"] / dataset_name

    evaluator = SLAMEvaluator(dataset_path)

    results = evaluator.run_and_evaluate_all(
        dataset_formats=[args.dataset_format],
        sensor_types=[args.sensor_type]
    )

    baseline_results_dir = dirs["results_dir"] / "baseline"
    baseline_results_dir.mkdir(exist_ok=True)

    with open(baseline_results_dir / "results.json", "w") as f:
        json.dump(results, f, indent=4)

    for src_file in (dataset_path / "results" / "comparison").glob("*.*"):
        dst_file = baseline_results_dir / src_file.name
        shutil.copy(src_file, dst_file)

    return results

def enhance_slam(args, dirs, model_paths=None, dataset_path=None):
    if dataset_path is None:
        scenario = args.scenario
        dataset_name = f"{scenario}_dataset"
        dataset_path = dirs["data_dir"] / dataset_name

    if model_paths is None:
        feature_model_dir = dirs["models_dir"] / "feature_enhancement"
        loop_closure_model_dir = dirs["models_dir"] / "loop_closure"
        patch_model_dir = dirs["models_dir"] / "patch_descriptor"

        model_paths = {
            "feature_enhancement": str(feature_model_dir / "feature_enhancement.onnx"),
            "loop_closure": str(loop_closure_model_dir / "loop_closure.onnx"),
            "descriptor": str(patch_model_dir / "patch_descriptor.onnx")
        }

    orb_slam_path = Path(args.orb_slam_path)

    enhanced_dataset_path = dataset_path / "preprocessed" / args.dataset_format

    enhancer = ORBSLAMEnhancer(orb_slam_path, model_paths)

    enhanced_dataset = enhancer.enhance_dataset(enhanced_dataset_path)

    enhanced_output_dir = dirs["results_dir"] / "enhanced"
    enhanced_output_dir.mkdir(exist_ok=True)

    result_path = enhancer.run_enhanced_orb_slam(
        dataset_format=args.dataset_format,
        dataset_path=enhanced_dataset,
        output_path=enhanced_output_dir,
        sensor_type=args.sensor_type
    )

    return result_path

def evaluate_enhanced_slam(args, dirs, enhanced_result_path=None):
    if enhanced_result_path is None:
        enhanced_result_path = dirs["results_dir"] / "enhanced"

    scenario = args.scenario
    dataset_name = f"{scenario}_dataset"
    dataset_path = dirs["data_dir"] / dataset_name

    evaluator = SLAMEvaluator(dataset_path)

    enhanced_trajectory = enhanced_result_path / "CameraTrajectory.txt"

    if args.dataset_format == "tum":
        gt_trajectory = dataset_path / "preprocessed" / "tum_format" / "groundtruth.txt"
    elif args.dataset_format == "euroc":
        gt_trajectory = dataset_path / "preprocessed" / "euroc_format" / "mav0" / "state_groundtruth_estimate0" / "data.csv"

        converted_gt_file = enhanced_result_path / "groundtruth.txt"
        evaluator._convert_euroc_gt_to_tum(gt_trajectory, converted_gt_file)
        gt_trajectory = converted_gt_file
    elif args.dataset_format == "kitti":
        gt_trajectory = dataset_path / "preprocessed" / "kitti_format" / "poses.txt"

        converted_gt_file = enhanced_result_path / "groundtruth.txt"
        evaluator._convert_kitti_gt_to_tum(gt_trajectory, converted_gt_file)
        gt_trajectory = converted_gt_file

    ate_results, ate_errors = evaluator.calculate_ate(gt_trajectory, enhanced_trajectory)
    rpe_results, trans_errors, rot_errors = evaluator.calculate_rpe(gt_trajectory, enhanced_trajectory)

    tracking_results = evaluator.calculate_tracking_success_rate(enhanced_result_path)

    results = {
        "algorithm": "enhanced_orb_slam3",
        "dataset_format": args.dataset_format,
        "sensor_type": args.sensor_type,
        "ate": ate_results,
        "rpe": rpe_results,
        "tracking": tracking_results
    }

    with open(enhanced_result_path / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)

    evaluator._plot_trajectory(gt_trajectory, enhanced_trajectory, enhanced_result_path / "trajectory_plot.png")
    evaluator._plot_ate_errors(ate_errors, enhanced_result_path / "ate_errors.png")
    evaluator._plot_rpe_errors(trans_errors, rot_errors, enhanced_result_path / "rpe_errors.png")

    return results

def compare_results(args, dirs, baseline_results=None, enhanced_results=None):
    comparison_dir = dirs["results_dir"] / "comparison"
    comparison_dir.mkdir(exist_ok=True)

    if baseline_results is None:
        baseline_results_file = dirs["results_dir"] / "baseline" / "results.json"

        if baseline_results_file.exists():
            with open(baseline_results_file, "r") as f:
                baseline_results = json.load(f)
        else:
            print("Baseline results not found.")
            return None

    if enhanced_results is None:
        enhanced_results_file = dirs["results_dir"] / "enhanced" / "evaluation_results.json"

        if enhanced_results_file.exists():
            with open(enhanced_results_file, "r") as f:
                enhanced_results = json.load(f)
        else:
            print("Enhanced results not found.")
            return None

    if isinstance(baseline_results, list):
        baseline = None

        for result in baseline_results:
            if result["algorithm"] == "orb_slam3" and result["sensor_type"] == args.sensor_type:
                baseline = result
                break

        if baseline is None:
            print("No matching baseline results found.")
            return None
    else:
        baseline = baseline_results

    enhanced = enhanced_results

    comparison = {
        "baseline": baseline,
        "enhanced": enhanced,
        "improvement": {
            "ate": {
                "rmse": baseline["ate"]["rmse"] - enhanced["ate"]["rmse"],
                "rmse_percent": (baseline["ate"]["rmse"] - enhanced["ate"]["rmse"]) / baseline["ate"]["rmse"] * 100 if baseline["ate"]["rmse"] > 0 else 0,
                "mean": baseline["ate"]["mean"] - enhanced["ate"]["mean"],
                "mean_percent": (baseline["ate"]["mean"] - enhanced["ate"]["mean"]) / baseline["ate"]["mean"] * 100 if baseline["ate"]["mean"] > 0 else 0
            },
            "rpe": {
                "translation": {
                    "rmse": baseline["rpe"]["translation"]["rmse"] - enhanced["rpe"]["translation"]["rmse"],
                    "rmse_percent": (baseline["rpe"]["translation"]["rmse"] - enhanced["rpe"]["translation"]["rmse"]) / baseline["rpe"]["translation"]["rmse"] * 100 if baseline["rpe"]["translation"]["rmse"] > 0 else 0
                },
                "rotation": {
                    "rmse": baseline["rpe"]["rotation"]["rmse"] - enhanced["rpe"]["rotation"]["rmse"],
                    "rmse_percent": (baseline["rpe"]["rotation"]["rmse"] - enhanced["rpe"]["rotation"]["rmse"]) / baseline["rpe"]["rotation"]["rmse"] * 100 if baseline["rpe"]["rotation"]["rmse"] > 0 else 0
                }
            },
            "tracking": {
                "success_rate": enhanced["tracking"]["tracking_success_rate"] - baseline["tracking"]["tracking_success_rate"],
                "success_rate_percent": (enhanced["tracking"]["tracking_success_rate"] - baseline["tracking"]["tracking_success_rate"]) / baseline["tracking"]["tracking_success_rate"] * 100 if baseline["tracking"]["tracking_success_rate"] > 0 else 0
            }
        }
    }

    with open(comparison_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=4)

    plt.figure(figsize=(12, 8))

    metrics = [
        ("ATE RMSE [m]", baseline["ate"]["rmse"], enhanced["ate"]["rmse"]),
        ("ATE Mean [m]", baseline["ate"]["mean"], enhanced["ate"]["mean"]),
        ("RPE Trans. RMSE [m]", baseline["rpe"]["translation"]["rmse"], enhanced["rpe"]["translation"]["rmse"]),
        ("RPE Rot. RMSE [°]", baseline["rpe"]["rotation"]["rmse"], enhanced["rpe"]["rotation"]["rmse"]),
        ("Tracking Success Rate", baseline["tracking"]["tracking_success_rate"], enhanced["tracking"]["tracking_success_rate"])
    ]

    x = np.arange(len(metrics))
    width = 0.35

    baseline_values = [m[1] for m in metrics]
    enhanced_values = [m[2] for m in metrics]

    plt.bar(x - width/2, baseline_values, width, label='Baseline ORB-SLAM3')
    plt.bar(x + width/2, enhanced_values, width, label='ML-Enhanced ORB-SLAM3')

    plt.xlabel('Metrics')
    plt.ylabel('Value')
    plt.title('Comparison of Baseline vs. ML-Enhanced ORB-SLAM3')
    plt.xticks(x, [m[0] for m in metrics], rotation=45)
    plt.legend()

    plt.tight_layout()
    plt.savefig(comparison_dir / "metrics_comparison.png")

    plt.figure(figsize=(12, 6))

    improvement_percentages = [
        comparison["improvement"]["ate"]["rmse_percent"],
        comparison["improvement"]["ate"]["mean_percent"],
        comparison["improvement"]["rpe"]["translation"]["rmse_percent"],
        comparison["improvement"]["rpe"]["rotation"]["rmse_percent"],
        comparison["improvement"]["tracking"]["success_rate_percent"]
    ]

    colors = ['green' if val > 0 else 'red' for val in improvement_percentages]

    plt.bar(x, improvement_percentages, color=colors)

    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Metrics')
    plt.ylabel('Improvement %')
    plt.title('Percentage Improvement of ML-Enhanced over Baseline ORB-SLAM3')
    plt.xticks(x, [m[0] for m in metrics], rotation=45)

    plt.tight_layout()
    plt.savefig(comparison_dir / "improvement_percentages.png")

    create_html_report(comparison, comparison_dir)

    return comparison

def create_html_report(comparison, output_dir):
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ML-Enhanced SLAM Evaluation Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: right; }
            th { background-color: #f2f2f2; }
            .header { text-align: left; }
            .improved { color: green; }
            .worsened { color: red; }
            .section { margin-top: 30px; }
            img { max-width: 100%; }
            .container { display: flex; flex-wrap: wrap; }
            .chart { width: 100%; margin-bottom: 20px; }
            @media (min-width: 768px) {
                .chart { width: 48%; margin-right: 2%; }
            }
        </style>
    </head>
    <body>
        <h1>ML-Enhanced SLAM Evaluation Report</h1>

        <div class="section">
            <h2>Performance Metrics Comparison</h2>
            <table>
                <tr>
                    <th class="header">Metric</th>
                    <th>Baseline ORB-SLAM3</th>
                    <th>ML-Enhanced ORB-SLAM3</th>
                    <th>Improvement</th>
                    <th>Improvement %</th>
                </tr>
                <tr>
                    <td class="header">ATE RMSE [m]</td>
                    <td>""" + f"{comparison['baseline']['ate']['rmse']:.4f}" + """</td>
                    <td>""" + f"{comparison['enhanced']['ate']['rmse']:.4f}" + """</td>
                    <td class='""" + ("improved" if comparison["improvement"]["ate"]["rmse"] > 0 else "worsened") + """'>""" + f"{comparison['improvement']['ate']['rmse']:.4f}" + """</td>
                    <td class='""" + ("improved" if comparison["improvement"]["ate"]["rmse_percent"] > 0 else "worsened") + """'>""" + f"{comparison['improvement']['ate']['rmse_percent']:.2f}%" + """</td>
                </tr>
                <tr>
                    <td class="header">ATE Mean [m]</td>
                    <td>""" + f"{comparison['baseline']['ate']['mean']:.4f}" + """</td>
                    <td>""" + f"{comparison['enhanced']['ate']['mean']:.4f}" + """</td>
                    <td class='""" + ("improved" if comparison["improvement"]["ate"]["mean"] > 0 else "worsened") + """'>""" + f"{comparison['improvement']['ate']['mean']:.4f}" + """</td>
                    <td class='""" + ("improved" if comparison["improvement"]["ate"]["mean_percent"] > 0 else "worsened") + """'>""" + f"{comparison['improvement']['ate']['mean_percent']:.2f}%" + """</td>
                </tr>
                <tr>
                    <td class="header">RPE Translation RMSE [m]</td>
                    <td>""" + f"{comparison['baseline']['rpe']['translation']['rmse']:.4f}" + """</td>
                    <td>""" + f"{comparison['enhanced']['rpe']['translation']['rmse']:.4f}" + """</td>
                    <td class='""" + ("improved" if comparison["improvement"]["rpe"]["translation"]["rmse"] > 0 else "worsened") + """'>""" + f"{comparison['improvement']['rpe']['translation']['rmse']:.4f}" + """</td>
                    <td class='""" + ("improved" if comparison["improvement"]["rpe"]["translation"]["rmse_percent"] > 0 else "worsened") + """'>""" + f"{comparison['improvement']['rpe']['translation']['rmse_percent']:.2f}%" + """</td>
                </tr>
                <tr>
                    <td class="header">RPE Rotation RMSE [°]</td>
                    <td>""" + f"{comparison['baseline']['rpe']['rotation']['rmse']:.4f}" + """</td>
                    <td>""" + f"{comparison['enhanced']['rpe']['rotation']['rmse']:.4f}" + """</td>
                    <td class='""" + ("improved" if comparison["improvement"]["rpe"]["rotation"]["rmse"] > 0 else "worsened") + """'>""" + f"{comparison['improvement']['rpe']['rotation']['rmse']:.4f}" + """</td>
                    <td class='""" + ("improved" if comparison["improvement"]["rpe"]["rotation"]["rmse_percent"] > 0 else "worsened") + """'>""" + f"{comparison['improvement']['rpe']['rotation']['rmse_percent']:.2f}%" + """</td>
                </tr>
                <tr>
                    <td class="header">Tracking Success Rate</td>
                    <td>""" + f"{comparison['baseline']['tracking']['tracking_success_rate']:.4f}" + """</td>
                    <td>""" + f"{comparison['enhanced']['tracking']['tracking_success_rate']:.4f}" + """</td>
                    <td class='""" + ("improved" if comparison["improvement"]["tracking"]["success_rate"] > 0 else "worsened") + """'>""" + f"{comparison['improvement']['tracking']['success_rate']:.4f}" + """</td>
                    <td class='""" + ("improved" if comparison["improvement"]["tracking"]["success_rate_percent"] > 0 else "worsened") + """'>""" + f"{comparison['improvement']['tracking']['success_rate_percent']:.2f}%" + """</td>
                </tr>
            </table>
        </div>

        <div class="section">
            <h2>Visualization</h2>
            <div class="container">
                <div class="chart">
                    <h3>Metrics Comparison</h3>
                    <img src="metrics_comparison.png" alt="Metrics Comparison">
                </div>
                <div class="chart">
                    <h3>Improvement Percentages</h3>
                    <img src="improvement_percentages.png" alt="Improvement Percentages">
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Trajectory Comparison</h2>
            <div class="container">
                <div class="chart">
                    <h3>Baseline Trajectory</h3>
                    <img src="../baseline/trajectory_plot.png" alt="Baseline Trajectory">
                </div>
                <div class="chart">
                    <h3>Enhanced Trajectory</h3>
                    <img src="../enhanced/trajectory_plot.png" alt="Enhanced Trajectory">
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Error Distribution</h2>
            <div class="container">
                <div class="chart">
                    <h3>Baseline ATE Errors</h3>
                    <img src="../baseline/ate_errors.png" alt="Baseline ATE Errors">
                </div>
                <div class="chart">
                    <h3>Enhanced ATE Errors</h3>
                    <img src="../enhanced/ate_errors.png" alt="Enhanced ATE Errors">
                </div>
            </div>
            <div class="container">
                <div class="chart">
                    <h3>Baseline RPE Errors</h3>
                    <img src="../baseline/rpe_errors.png" alt="Baseline RPE Errors">
                </div>
                <div class="chart">
                    <h3>Enhanced RPE Errors</h3>
                    <img src="../enhanced/rpe_errors.png" alt="Enhanced RPE Errors">
                </div>
            </div>
        </div>
    </body>
    </html>
    """

    with open(output_dir / "report.html", "w") as f:
        f.write(html)

def full_pipeline(args, dirs):
    print("=== Running Full Pipeline ===")

    print("\n1. Running Simulation...")
    dataset_path = run_simulation(args, dirs)

    print("\n2. Preprocessing Data...")
    preprocessed_paths = preprocess_data(args, dirs, dataset_path)

    print("\n3. Training ML Models...")
    ml_dataset_path = dataset_path / "preprocessed" / "ml_dataset"
    model_paths = train_ml_models(args, dirs, ml_dataset_path)

    print("\n4. Evaluating Baseline SLAM...")
    baseline_results = evaluate_baseline_slam(args, dirs, dataset_path)

    print("\n5. Enhancing SLAM with ML...")
    enhanced_result_path = enhance_slam(args, dirs, model_paths, dataset_path)

    print("\n6. Evaluating Enhanced SLAM...")
    enhanced_results = evaluate_enhanced_slam(args, dirs, enhanced_result_path)

    print("\n7. Comparing Results...")
    comparison = compare_results(args, dirs, baseline_results, enhanced_results)

    print("\nFull Pipeline Complete!")
    print(f"Results saved to: {dirs['results_dir']}")

    return comparison

def main():
    args = parse_args()
    dirs = setup_environment(args)

    if args.mode == "simulate":
        dataset_path = run_simulation(args, dirs)
        print(f"Simulation completed. Dataset saved to: {dataset_path}")
    elif args.mode == "preprocess":
        output_paths = preprocess_data(args, dirs)
        print(f"Preprocessing completed. Outputs saved to: {output_paths}")
    elif args.mode == "train_ml":
        model_paths = train_ml_models(args, dirs)
        print(f"ML model training completed. Models saved to: {dirs['models_dir']}")
    elif args.mode == "evaluate_baseline":
        results = evaluate_baseline_slam(args, dirs)
        print(f"Baseline evaluation completed. Results saved to: {dirs['results_dir'] / 'baseline'}")
    elif args.mode == "enhance_slam":
        result_path = enhance_slam(args, dirs)
        print(f"SLAM enhancement completed. Results saved to: {result_path}")
    elif args.mode == "evaluate_enhanced":
        results = evaluate_enhanced_slam(args, dirs)
        print(f"Enhanced SLAM evaluation completed. Results saved to: {dirs['results_dir'] / 'enhanced'}")
    elif args.mode == "full_pipeline":
        comparison = full_pipeline(args, dirs)
        print(f"Full pipeline completed. Final comparison report: {dirs['results_dir'] / 'comparison' / 'report.html'}")
    else:
        print(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()

    print(f"Total execution time: {end_time - start_time:.2f} seconds")
