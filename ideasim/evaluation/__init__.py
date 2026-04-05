"""Evaluation framework for the Ideological Dynamics Simulator.

Provides dataset loading, parameter fitting, temporal cross-validation,
and proper ML-style metrics for validating the simulation engine against
historical ideology share data.
"""

from ideasim.evaluation.dataset import ObservedDataPoint, ObservedDataset
from ideasim.evaluation.fitting import fit_parameters
from ideasim.evaluation.metrics import EvaluationMetrics, compute_metrics
from ideasim.evaluation.runner import run_evaluation
from ideasim.evaluation.splits import temporal_kfold, temporal_train_test_split

__all__ = [
    "EvaluationMetrics",
    "ObservedDataPoint",
    "ObservedDataset",
    "compute_metrics",
    "fit_parameters",
    "run_evaluation",
    "temporal_kfold",
    "temporal_train_test_split",
]
