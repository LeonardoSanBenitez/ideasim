"""Temporal splitting strategies for evaluation.

All splits respect temporal ordering to prevent data leakage:
- Training always uses earlier years than validation.
- No shuffling across time boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

from ideasim.evaluation.dataset import ObservedDataset


@dataclass(frozen=True)
class TemporalSplit:
    """A single train/validation split defined by year boundaries.

    Training: [train_start, train_end]
    Validation: (train_end, val_end]
    """

    train_start: float
    train_end: float
    val_end: float
    fold_index: int = 0

    @property
    def val_start(self) -> float:
        """Validation starts right after training ends."""
        return self.train_end

    def split_dataset(
        self, dataset: ObservedDataset
    ) -> tuple[ObservedDataset, ObservedDataset]:
        """Split a dataset into train and validation subsets.

        Training includes [train_start, train_end].
        Validation includes (train_end, val_end] (exclusive on left).
        """
        train = dataset.filter_years(self.train_start, self.train_end)
        val = dataset.filter_years_exclusive_start(self.train_end, self.val_end)
        return train, val


def temporal_train_test_split(
    dataset: ObservedDataset,
    train_fraction: float = 0.7,
) -> TemporalSplit:
    """Single temporal train/test split.

    The first `train_fraction` of unique years go to training,
    the rest to validation.

    Args:
        dataset: The observed dataset to split.
        train_fraction: Fraction of years for training (default 0.7).

    Returns:
        A TemporalSplit defining the boundary.
    """
    years = dataset.years
    if len(years) < 2:
        raise ValueError(
            f"Dataset has {len(years)} unique year(s), need at least 2 for splitting."
        )

    split_idx = max(1, int(len(years) * train_fraction))
    split_idx = min(split_idx, len(years) - 1)  # ensure at least 1 val year

    train_end = float(years[split_idx - 1])
    val_end = float(years[-1])

    return TemporalSplit(
        train_start=float(years[0]),
        train_end=train_end,
        val_end=val_end,
        fold_index=0,
    )


def temporal_kfold(
    dataset: ObservedDataset,
    n_folds: int = 5,
    min_train_fraction: float = 0.3,
) -> list[TemporalSplit]:
    """Time-series k-fold cross-validation (expanding window).

    Each fold uses an expanding training window and a fixed-size
    validation window. This prevents data leakage: training always
    precedes validation temporally.

    Fold 0: train on years[0..s1], validate on years[s1..s2]
    Fold 1: train on years[0..s2], validate on years[s2..s3]
    ...
    Fold k-1: train on years[0..sk], validate on years[sk..end]

    This is the standard "expanding window" approach used in
    time-series cross-validation (sklearn's TimeSeriesSplit).

    Args:
        dataset: The observed dataset.
        n_folds: Number of folds.
        min_train_fraction: Minimum fraction of years for the first
            training set (default 0.3).

    Returns:
        List of TemporalSplit objects, one per fold.
    """
    years = dataset.years
    n_years = len(years)

    if n_years < n_folds + 1:
        raise ValueError(
            f"Dataset has {n_years} unique years, need at least {n_folds + 1} "
            f"for {n_folds}-fold temporal CV."
        )

    min_train_size = max(1, int(n_years * min_train_fraction))

    # Divide the remaining years into n_folds validation windows
    remaining = n_years - min_train_size
    if remaining < n_folds:
        raise ValueError(
            f"Not enough years ({remaining} available) for {n_folds} folds "
            f"with min_train_fraction={min_train_fraction}."
        )

    fold_size = remaining // n_folds

    splits: list[TemporalSplit] = []
    for fold in range(n_folds):
        train_end_idx = min_train_size + fold * fold_size - 1
        if fold < n_folds - 1:
            val_end_idx = min_train_size + (fold + 1) * fold_size - 1
        else:
            # Last fold gets all remaining years
            val_end_idx = n_years - 1

        splits.append(
            TemporalSplit(
                train_start=float(years[0]),
                train_end=float(years[train_end_idx]),
                val_end=float(years[val_end_idx]),
                fold_index=fold,
            )
        )

    return splits
