"""
NAVRIS Phase 3.0: Machine Learning Module for Navigation Residual Correction.
"""

from navris.ml.dataset import load_recording_ml_data, build_ml_dataset, RecordingSplit
from navris.ml.features import CausalFeatureExtractor, FEATURE_NAMES
from navris.ml.model import VelocityResidualXGBoost
from navris.ml.evaluate import evaluate_residual_predictions, format_metrics_table
from navris.ml.leakage_audit import run_leakage_audit

__all__ = [
    "load_recording_ml_data",
    "build_ml_dataset",
    "RecordingSplit",
    "CausalFeatureExtractor",
    "FEATURE_NAMES",
    "VelocityResidualXGBoost",
    "evaluate_residual_predictions",
    "format_metrics_table",
    "run_leakage_audit",
]
