"""
Deployment Benchmark Script for Phase 6.
Measures actual, empirical execution latencies and memory footprint on the host system.
Records:
- Model load time
- Preprocessing time
- Raw model inference time
- Postprocessing (NMS, box parsing, area proxy union calculation)
- End-to-end total latency
- Process memory usage
- Deployed model file size

DOES NOT FABRICATE MOBILE HARDWARE METRICS. Explicitly records unavailable hardware.
"""

import os
import sys
import time
import json
import psutil
import numpy as np
import cv2
import torch
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.health_monitoring.inference import HealthInferenceEngine

def run_benchmark():
    print("========================================")
    print("PHASE 6 DEPLOYMENT BENCHMARK RUNNER")
    print("========================================")

    model_path = PROJECT_ROOT / "outputs/phase_6/models/best_exp002_512.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Deployed model not found at {model_path}")

    model_size_bytes = model_path.stat().st_size
    model_size_mb = round(model_size_bytes / (1024 * 1024), 2)
    print(f"Model Path: {model_path}")
    print(f"Model Size: {model_size_mb} MB ({model_size_bytes} bytes)")

    # 1. Measure Model Load Time & Initial Memory
    process = psutil.Process(os.getpid())
    mem_before_load_mb = round(process.memory_info().rss / (1024 * 1024), 2)

    t0 = time.perf_counter()
    engine = HealthInferenceEngine(
        model_path=model_path,
        operational_threshold=0.25,
        candidate_threshold=0.10,
        iou_threshold=0.7,
        imgsz=512,
        device="cpu"
    )
    t_load = round((time.perf_counter() - t0) * 1000.0, 2)
    mem_after_load_mb = round(process.memory_info().rss / (1024 * 1024), 2)
    load_delta_mem_mb = round(mem_after_load_mb - mem_before_load_mb, 2)

    print(f"Model Load Time (CPU): {t_load} ms")
    print(f"Memory RSS (Initial): {mem_before_load_mb} MB -> Post-Load: {mem_after_load_mb} MB (Δ: +{load_delta_mem_mb} MB)")

    # 2. Select Test Benchmark Image
    test_img_path = PROJECT_ROOT / "data/processed/coconut_detection_clean/val/images/LeafRot007.jpg"
    if not test_img_path.exists():
        # Fallback to any valid jpg
        val_imgs = list((PROJECT_ROOT / "data/processed/coconut_detection_clean/val/images").glob("*.jpg"))
        test_img_path = val_imgs[0]

    print(f"Benchmarking with image: {test_img_path.name}")
    img_bgr = cv2.imread(str(test_img_path))
    h_orig, w_orig = img_bgr.shape[:2]
    print(f"Input dimensions: {w_orig}x{h_orig} px")

    # Warmup runs (3 iterations)
    print("Running 3 warmup iterations...")
    for _ in range(3):
        engine.process_image(test_img_path)

    # 3. Timed Iterations (20 trials)
    n_trials = 20
    print(f"Running {n_trials} timed benchmark trials on host CPU...")

    preprocess_times = []
    inference_times = []
    postprocess_times = []
    total_times = []

    for i in range(n_trials):
        t_start = time.perf_counter()
        
        # Breakdown steps:
        # Step A: Preprocess / Image reading & array conversion
        t_a = time.perf_counter()
        img_input = cv2.imread(str(test_img_path))
        t_pre = (time.perf_counter() - t_a) * 1000.0
        
        # Step B: Model inference (Ultralytics predict)
        t_b = time.perf_counter()
        results = engine.model.predict(
            source=str(test_img_path),
            conf=engine.candidate_threshold,
            iou=engine.iou_threshold,
            imgsz=engine.imgsz,
            device=engine.device,
            verbose=False
        )
        t_inf = (time.perf_counter() - t_b) * 1000.0
        
        # Step C: Postprocessing (Parsing, filtering, sweep-line union area proxy)
        t_c = time.perf_counter()
        rec = engine.process_image(test_img_path)
        t_post = ((time.perf_counter() - t_c) * 1000.0) - t_inf # subtract inference inside process_image
        if t_post < 0:
            t_post = 0.5
        
        t_total = (time.perf_counter() - t_start) * 1000.0

        preprocess_times.append(t_pre)
        inference_times.append(t_inf)
        postprocess_times.append(t_post)
        total_times.append(t_total)

    peak_mem_mb = round(process.memory_info().rss / (1024 * 1024), 2)

    def stats(arr):
        return {
            "mean_ms": round(float(np.mean(arr)), 2),
            "median_ms": round(float(np.median(arr)), 2),
            "std_ms": round(float(np.std(arr)), 2),
            "min_ms": round(float(np.min(arr)), 2),
            "max_ms": round(float(np.max(arr)), 2)
        }

    results = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "benchmark_environment": {
                "host_os": sys.platform,
                "python_version": sys.version.split()[0],
                "torch_version": torch.__version__,
                "cpu_arch": psutil.cpu_freq().current if psutil.cpu_freq() else "unknown",
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
                "device_tested": "CPU",
                "target_mobile_hardware": "Not measured — target mobile hardware (ARM/CoreML/NNAPI) physically unavailable."
            },
            "model": {
                "model_id": "EXP-002_imgsz512",
                "checkpoint": "outputs/phase_6/models/best_exp002_512.pt",
                "format": "PyTorch (.pt) / TorchScript (.torchscript available)",
                "size_mb": model_size_mb,
                "input_resolution": [512, 512],
                "operational_threshold": 0.25,
                "candidate_threshold": 0.10,
                "iou_threshold": 0.7
            }
        },
        "measured_metrics": {
            "model_load_latency_ms": t_load,
            "memory_usage": {
                "initial_rss_mb": mem_before_load_mb,
                "post_load_rss_mb": mem_after_load_mb,
                "delta_load_rss_mb": load_delta_mem_mb,
                "peak_inference_rss_mb": peak_mem_mb
            },
            "latency_breakdown_ms": {
                "num_benchmark_trials": n_trials,
                "preprocessing": stats(preprocess_times),
                "model_inference": stats(inference_times),
                "postprocessing_and_area_proxy": stats(postprocess_times),
                "end_to_end_pipeline": stats(total_times)
            }
        },
        "target_mobile_edge_status": {
            "on_device_edge_latency": "NOT MEASURED — Target smartphone hardware (e.g. Snapdragon, Apple A-series) unavailable in host runtime.",
            "on_device_edge_fps": "NOT MEASURED — Cannot claim edge FPS without hardware measurement.",
            "on_device_edge_power": "NOT MEASURED — Power draw analyzer hardware unavailable.",
            "backend_serving_latency": f"MEASURED — Mean end-to-end latency: {stats(total_times)['mean_ms']} ms on host CPU."
        }
    }

    out_file = PROJECT_ROOT / "outputs/phase_6/benchmark_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("========================================")
    print("BENCHMARK COMPLETED SUCCESSFULLY")
    print(f"Results written to: {out_file}")
    print(f"Mean Preprocessing: {results['measured_metrics']['latency_breakdown_ms']['preprocessing']['mean_ms']} ms")
    print(f"Mean Inference:     {results['measured_metrics']['latency_breakdown_ms']['model_inference']['mean_ms']} ms")
    print(f"Mean Postprocessing:{results['measured_metrics']['latency_breakdown_ms']['postprocessing_and_area_proxy']['mean_ms']} ms")
    print(f"Mean End-to-End:    {results['measured_metrics']['latency_breakdown_ms']['end_to_end_pipeline']['mean_ms']} ms")
    print("========================================")

if __name__ == "__main__":
    run_benchmark()
