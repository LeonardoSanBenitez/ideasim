"""ParlGov data ingestion: convert ParlGov CSV files to ObservedDataset.

Downloads and processes election results and party data from the ParlGov
project (https://www.parlgov.org/). Aggregates parliamentary vote shares
by party family, then maps party families to the simulation's ideology
categories.

The mapping is configurable but ships with a sensible default for the
standard 4-ideology European scenario.
"""

from __future__ import annotations

import csv
import logging
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

from ideasim.evaluation.dataset import ObservedDataPoint, ObservedDataset

logger = logging.getLogger(__name__)

# ParlGov development CSV URLs (stable, updated regularly)
PARLGOV_ELECTION_URL = (
    "https://www.parlgov.org/data/parlgov-development_csv-utf-8/view_election.csv"
)
PARLGOV_PARTY_URL = (
    "https://www.parlgov.org/data/parlgov-development_csv-utf-8/view_party.csv"
)

# Default set of European countries (ISO 3166-1 alpha-3 as used by ParlGov)
DEFAULT_EUROPEAN_COUNTRIES: frozenset[str] = frozenset(
    {
        "AUT",
        "BEL",
        "CHE",
        "CZE",
        "DEU",
        "DNK",
        "ESP",
        "FIN",
        "FRA",
        "GBR",
        "GRC",
        "HUN",
        "IRL",
        "ISL",
        "ITA",
        "LUX",
        "NLD",
        "NOR",
        "POL",
        "PRT",
        "SVK",
        "SVN",
        "SWE",
    }
)

# Default mapping from ParlGov party family names to simulation ideologies.
# Parties whose family maps to None are excluded from the dataset.
DEFAULT_FAMILY_MAPPING: dict[str, str | None] = {
    "Liberal": "liberalism",
    "Green/Ecologist": "liberalism",
    "Social democracy": "marxism",
    "Communist/Socialist": "marxism",
    "Conservative": "traditionalism",
    "Christian democracy": "traditionalism",
    "Agrarian": "traditionalism",
    "Right-wing": "nationalism",
    "Special issue": None,
    "no family": None,
    "to be coded": None,
}


def download_csv(url: str, dest: Path) -> Path:
    """Download a CSV file from a URL if it doesn't already exist.

    Args:
        url: Remote URL of the CSV file.
        dest: Local file path to save the download.

    Returns:
        The path to the downloaded file.
    """
    if dest.exists():
        logger.info("Using cached file: %s", dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s", url, dest)
    urllib.request.urlretrieve(url, dest)
    return dest


def _load_party_families(party_csv: Path) -> dict[str, str]:
    """Load party_id -> family_name mapping from ParlGov party CSV.

    Args:
        party_csv: Path to the ParlGov view_party.csv file.

    Returns:
        Dict mapping party_id (as string) to family_name.
    """
    party_family: dict[str, str] = {}
    with open(party_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            party_family[row["party_id"]] = row["family_name"]
    return party_family


def _aggregate_vote_shares(
    election_csv: Path,
    party_families: dict[str, str],
    family_mapping: dict[str, str | None],
    countries: frozenset[str],
    decade_aggregation: bool = True,
) -> dict[int, dict[str, float]]:
    """Aggregate vote shares by ideology per time period.

    Reads the ParlGov election CSV, filters to parliamentary elections in
    the specified countries, maps each party to an ideology via its family,
    and aggregates vote shares.

    Args:
        election_csv: Path to ParlGov view_election.csv.
        party_families: party_id -> family_name mapping.
        family_mapping: family_name -> ideology name (or None to skip).
        countries: Set of country codes to include.
        decade_aggregation: If True, aggregate by decade. If False,
            aggregate by individual election year.

    Returns:
        Dict mapping year (or decade start year) to {ideology: share}.
        Shares are normalized to sum to 1.0 per period.
    """
    # Raw vote totals: period -> ideology -> cumulative vote_share
    raw: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    totals: dict[int, float] = defaultdict(float)

    with open(election_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["election_type"] != "parliament":
                continue
            if row["country_name_short"] not in countries:
                continue
            if not row["vote_share"]:
                continue

            vote_share = float(row["vote_share"])
            year = int(row["election_date"][:4])
            period = (year // 10) * 10 if decade_aggregation else year

            party_id = row["party_id"]
            family = party_families.get(party_id)
            if family is None:
                continue
            ideology = family_mapping.get(family)
            if ideology is None:
                continue

            raw[period][ideology] += vote_share
            totals[period] += vote_share

    # Normalize to shares that sum to 1.0
    result: dict[int, dict[str, float]] = {}
    for period in sorted(raw.keys()):
        total = totals[period]
        if total <= 0:
            continue
        result[period] = {
            ideo: round(raw_val / total, 6)
            for ideo, raw_val in sorted(raw[period].items())
        }

    return result


def build_parlgov_dataset(
    election_csv: Path,
    party_csv: Path,
    countries: frozenset[str] | None = None,
    family_mapping: dict[str, str | None] | None = None,
    decade_aggregation: bool = True,
    dataset_name: str = "parlgov_europe",
) -> ObservedDataset:
    """Build an ObservedDataset from ParlGov CSV files.

    Args:
        election_csv: Path to ParlGov view_election.csv.
        party_csv: Path to ParlGov view_party.csv.
        countries: Country codes to include. Defaults to Western/Central Europe.
        family_mapping: Party family -> ideology mapping. Defaults to the
            standard 4-ideology mapping.
        decade_aggregation: If True, aggregate by decade (recommended for
            the simulation's coarse time resolution).
        dataset_name: Name for the resulting dataset.

    Returns:
        ObservedDataset with normalized ideology shares per time period.
    """
    if countries is None:
        countries = DEFAULT_EUROPEAN_COUNTRIES
    if family_mapping is None:
        family_mapping = DEFAULT_FAMILY_MAPPING

    party_families = _load_party_families(party_csv)
    shares_by_period = _aggregate_vote_shares(
        election_csv=election_csv,
        party_families=party_families,
        family_mapping=family_mapping,
        countries=countries,
        decade_aggregation=decade_aggregation,
    )

    # Collect all ideology names that appear
    all_ideologies: set[str] = set()
    for period_shares in shares_by_period.values():
        all_ideologies.update(period_shares.keys())
    ideology_names = sorted(all_ideologies)

    # Build observations
    observations: list[ObservedDataPoint] = []
    for period, period_shares in sorted(shares_by_period.items()):
        for ideology in ideology_names:
            share = period_shares.get(ideology, 0.0)
            observations.append(
                ObservedDataPoint(
                    year=float(period),
                    ideology=ideology,
                    share=share,
                )
            )

    aggregation_mode = "decade" if decade_aggregation else "per-election-year"
    n_countries = len(countries)

    description = (
        f"European parliamentary vote shares aggregated by {aggregation_mode}, "
        f"derived from ParlGov data. Covers {n_countries} countries. "
        f"Party families mapped to 4-ideology system: "
        f"{', '.join(ideology_names)}."
    )

    metadata: dict[str, Any] = {
        "source": "ParlGov (https://www.parlgov.org/)",
        "aggregation": aggregation_mode,
        "countries": sorted(countries),
        "family_mapping": {k: v for k, v in family_mapping.items() if v is not None},
        "n_elections_processed": sum(
            1 for _ in _iter_election_rows(election_csv, countries)
        ),
    }

    return ObservedDataset(
        name=dataset_name,
        description=description,
        ideology_names=ideology_names,
        observations=observations,
        ideology_mapping={},  # names already match simulation
        metadata=metadata,
    )


def _iter_election_rows(
    election_csv: Path,
    countries: frozenset[str],
) -> Any:
    """Iterate over parliamentary election rows for given countries.

    Yields csv rows for counting purposes.
    """
    with open(election_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (
                row["election_type"] == "parliament"
                and row["country_name_short"] in countries
                and row["vote_share"]
            ):
                yield row


def download_and_build(
    cache_dir: Path,
    countries: frozenset[str] | None = None,
    family_mapping: dict[str, str | None] | None = None,
    decade_aggregation: bool = True,
    dataset_name: str = "parlgov_europe",
) -> ObservedDataset:
    """Download ParlGov data and build an ObservedDataset in one step.

    Args:
        cache_dir: Directory to cache downloaded CSV files.
        countries: Country codes to include.
        family_mapping: Party family -> ideology mapping.
        decade_aggregation: Aggregate by decade if True.
        dataset_name: Name for the dataset.

    Returns:
        ObservedDataset ready for use with the evaluation framework.
    """
    election_csv = download_csv(PARLGOV_ELECTION_URL, cache_dir / "view_election.csv")
    party_csv = download_csv(PARLGOV_PARTY_URL, cache_dir / "view_party.csv")

    return build_parlgov_dataset(
        election_csv=election_csv,
        party_csv=party_csv,
        countries=countries,
        family_mapping=family_mapping,
        decade_aggregation=decade_aggregation,
        dataset_name=dataset_name,
    )
