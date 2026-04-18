import os
import time
import json
import logging
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO

# logging configuration
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model = YOLO(self.config['model']['path'])
        self.device = torch.device(("cuda" if torch.cuda.is_available() else "cpu"))
        logger.info(f"Model loaded onto {self.device}")

    def benchmark_latency(self):
        """Measures precise GPU altency using torch.cuda.Event."""
        logger.info("Starting latency benchmark...")

        sample_dir = Path(self.config['paths']['inference_samples'])
        # Get a sample image path
        sample_img = str(next(sample_dir.glob("*.jpg")))

        for _ in range(20):
            _ = self.model(sample_img, verbose=False)
        
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        latencies = []

        with torch.no_grad():
            for _ in range(100):
                starter.record()
                _ = self.model(sample_img, verbose=False)
                ender.record()
                torch.cuda.synchronize()
                latencies.append(starter.elapsed_time(ender))

        avg_latency = sum(latencies) / len(latencies)
        logger.info(f"Average latency: {avg_latency:.2f}ms per frame")
        return avg_latency

    def run_validation(self):
        """Runs formal mAP validation"""
        logger.info("Running formal dataset validation...")
        metrics = self.model.val(data=self.config['paths']['test_data'], verbose=False)
        return metrics.results_dict

    def export_report(self, latency, metrics):
        """Saves a JSON report for CI/CD ingestion"""
        report = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'model_path': self.config['model']['path'],
            'metrics': {
                'mAP50': metrics['metrics/mAP50(B)'],
                'mAP50-95': metrics['metrics/mAP50-95(B)'],
                'latency_ms': latency,
                'fps': 1000 / latency if latency > 0 else float('inf'),
                'precision': metrics['metrics/precision(B)'],
                'recall': metrics['metrics/recall(B)']
            }
        }

        output_path = Path(self.config['paths']['report_output'])
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=4)
        logger.info(f"Report saved to {output_path}")
        return report

if __name__ == "__main__":
    evaluator = ModelEvaluator("configs/plate_detector_eval.yaml")
    latency = evaluator.benchmark_latency()
    metrics = evaluator.run_validation()
    evaluator.export_report(latency, metrics)