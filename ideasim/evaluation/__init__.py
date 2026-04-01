"""Evaluation framework for the Ideological Dynamics Simulator.

Provides dataset loading, parameter fitting, temporal cross-validation,
and proper ML-style metrics for validating the simulation engine against
historical ideology share data.
"""

from ideasim.evaluation.dataset import ObservedDataset, ObservedDataPoint
from ideasim.evaluation.metrics import compute_metrics, EvaluationMetrics
from ideasim.evaluation.splits import temporal_train_test_split, temporal_kfold
from ideasim.evaluation.fitting import fit_parameters
from ideasim.evaluation.runner import run_evaluation

__all__ = [
    "ObservedDataset",
    "ObservedDataPoint",
    "compute_metrics",
    "EvaluationMetrics",
    "temporal_train_test_split",
    "temporal_kfold",
    "fit_parameters",
    "run_evaluation",
]
