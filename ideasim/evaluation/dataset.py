"""Dataset abstraction for observed ideology share data.

A dataset is a collection of (year, ideology_name, share) observations.
Datasets are loaded from JSON files with a simple, documented schema.
This allows plugging in multiple data sources (ParlGov, MARPOR, custom)
without changing any evaluation code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ObservedDataPoint:
    """A single observation: ideology share at a given year."""

    year: float
    ideology: str
    share: float


@dataclass
class ObservedDataset:
    """A collection of observed ideology share data over time.

    Attributes:
        name: Human-readable dataset name (e.g. "parlgov_europe").
        description: Free-text description of the data source.
        ideology_names: Canonical ideology names used in this dataset.
        ideology_mapping: Maps dataset ideology names to simulation ideology
            names. E.g. {"social_democracy": "marxism"}. If empty, names
            are assumed to match the simulation directly.
        observations: The raw data points.
        metadata: Arbitrary metadata (source URL, version, etc.).
    """

    name: str
    description: str
    ideology_names: list[str]
    observations: list[ObservedDataPoint]
    ideology_mapping: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def years(self) -> NDArray[np.float64]:
        """Sorted unique years in the dataset."""
        unique = sorted(set(obs.year for obs in self.observations))
        return np.array(unique, dtype=np.float64)

    @property
    def year_range(self) -> tuple[float, float]:
        """(min_year, max_year) in the dataset."""
        years = self.years
        return float(years[0]), float(years[-1])

    def shares_at_year(self, year: float) -> dict[str, float]:
        """Return ideology -> share mapping for a specific year."""
        result: dict[str, float] = {}
        for obs in self.observations:
            if obs.year == year:
                mapped_name = self.ideology_mapping.get(
                    obs.ideology, obs.ideology
                )
                # Accumulate in case multiple dataset ideologies map
                # to the same simulation ideology
                result[mapped_name] = result.get(mapped_name, 0.0) + obs.share
        return result

    def as_matrix(
        self, sim_ideology_names: list[str]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Convert to (years, shares_matrix) aligned with simulation ideologies.

        Returns:
            years: shape (T,) sorted unique years.
            shares: shape (T, N) where N = len(sim_ideology_names).
                Missing ideologies get NaN so metrics can ignore them.
        """
        years = self.years
        n_ideo = len(sim_ideology_names)
        shares = np.full((len(years), n_ideo), np.nan, dtype=np.float64)

        for t_idx, year in enumerate(years):
            year_shares = self.shares_at_year(year)
            for i, name in enumerate(sim_ideology_names):
                if name in year_shares:
                    shares[t_idx, i] = year_shares[name]

        return years, shares

    def filter_years(
        self, min_year: float, max_year: float
    ) -> ObservedDataset:
        """Return a new dataset with only observations in [min_year, max_year]."""
        filtered = [
            obs for obs in self.observations
            if min_year <= obs.year <= max_year
        ]
        return ObservedDataset(
            name=self.name,
            description=self.description,
            ideology_names=self.ideology_names,
            observations=filtered,
            ideology_mapping=self.ideology_mapping,
            metadata=self.metadata,
        )

    def filter_years_exclusive_start(
        self, min_year: float, max_year: float
    ) -> ObservedDataset:
        """Return a new dataset with only observations in (min_year, max_year].

        The min_year boundary is exclusive to prevent overlap with a
        preceding training split that uses [start, min_year].
        """
        filtered = [
            obs for obs in self.observations
            if min_year < obs.year <= max_year
        ]
        return ObservedDataset(
            name=self.name,
            description=self.description,
            ideology_names=self.ideology_names,
            observations=filtered,
            ideology_mapping=self.ideology_mapping,
            metadata=self.metadata,
        )


def load_dataset(path: str | Path) -> ObservedDataset:
    """Load a dataset from a JSON file.

    Expected JSON schema:
    {
        "name": "dataset_name",
        "description": "...",
        "ideology_names": ["liberalism", "marxism", ...],
        "ideology_mapping": {"dataset_name": "sim_name", ...},  // optional
        "metadata": {...},  // optional
        "observations": [
            {"year": 1950.0, "ideology": "liberalism", "share": 0.35},
            ...
        ]
    }
    """
    with open(path) as f:
        data = json.load(f)

    observations = [
        ObservedDataPoint(
            year=float(obs["year"]),
            ideology=str(obs["ideology"]),
            share=float(obs["share"]),
        )
        for obs in data["observations"]
    ]

    return ObservedDataset(
        name=data["name"],
        description=data.get("description", ""),
        ideology_names=data["ideology_names"],
        observations=observations,
        ideology_mapping=data.get("ideology_mapping", {}),
        metadata=data.get("metadata", {}),
    )


def save_dataset(dataset: ObservedDataset, path: str | Path) -> None:
    """Save a dataset to JSON."""
    data: dict[str, Any] = {
        "name": dataset.name,
        "description": dataset.description,
        "ideology_names": dataset.ideology_names,
        "ideology_mapping": dataset.ideology_mapping,
        "metadata": dataset.metadata,
        "observations": [
            {
                "year": obs.year,
                "ideology": obs.ideology,
                "share": obs.share,
            }
            for obs in dataset.observations
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
