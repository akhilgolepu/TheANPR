import pytest
import json
import yaml
from pathlib import Path

def load_config():
    with open("configs/plate_detector_eval.yaml", "r") as f:
        return yaml.safe_load(f)

def load_report():
    with open("reports/plate_detector_report.json", "r") as f:
        return json.load(f)
    
def test_model_accuracy_threshold():
    """Test if the model meets minimum mAP targets."""
    config = load_config()
    report = load_report()
    
    actual_map = report['metrics']['mAP50']
    target_map = config['targets']['min_map50']
    
    assert actual_map >= target_map, f"Accuracy regression: {actual_map} < {target_map}"

def test_inference_speed_threshold():
    """Test if the model meets minimum inference speed targets."""
    config = load_config()
    report = load_report()
    
    actual_latency = report['metrics']['latency_ms']
    target_latency = config['targets']['max_latency_ms']
    
    assert actual_latency <= target_latency, f"Inference too slow: {actual_latency} > {target_latency}"

def test_model_file_exists():
    """Ensure the model weights were actaully saved"""
    config = load_config()
    assert Path(config['model']['path']).exists(), "Weights file is missing"