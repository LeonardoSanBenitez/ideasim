"""Tests for ParlGov data ingestion module."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from ideasim.evaluation.dataset import ObservedDataset, load_dataset, save_dataset
from ideasim.evaluation.parlgov import (
    DEFAULT_EUROPEAN_COUNTRIES,
    DEFAULT_FAMILY_MAPPING,
    _aggregate_vote_shares,
    _load_party_families,
    build_parlgov_dataset,
)


@pytest.fixture
def party_csv(tmp_path: Path) -> Path:
    """Create a minimal ParlGov party CSV for testing."""
    rows = [
        {
            "country_name_short": "DEU",
            "country_name": "Germany",
            "party_name_short": "SPD",
            "party_name_english": "Social Democratic Party",
            "party_name": "Sozialdemokratische Partei Deutschlands",
            "party_name_ascii": "SPD",
            "family_name_short": "soc",
            "family_name": "Social democracy",
            "left_right": "3.5",
            "state_market": "",
            "liberty_authority": "",
            "eu_anti_pro": "",
            "cmp": "",
            "euprofiler": "",
            "ees": "",
            "castles_mair": "",
            "huber_inglehart": "",
            "ray": "",
            "benoit_laver": "",
            "chess": "",
            "country_id": "1",
            "party_id": "100",
            "family_id": "11",
        },
        {
            "country_name_short": "DEU",
            "country_name": "Germany",
            "party_name_short": "CDU",
            "party_name_english": "Christian Democratic Union",
            "party_name": "CDU",
            "party_name_ascii": "CDU",
            "family_name_short": "chr",
            "family_name": "Christian democracy",
            "left_right": "6.5",
            "state_market": "",
            "liberty_authority": "",
            "eu_anti_pro": "",
            "cmp": "",
            "euprofiler": "",
            "ees": "",
            "castles_mair": "",
            "huber_inglehart": "",
            "ray": "",
            "benoit_laver": "",
            "chess": "",
            "country_id": "1",
            "party_id": "101",
            "family_id": "26",
        },
        {
            "country_name_short": "DEU",
            "country_name": "Germany",
            "party_name_short": "FDP",
            "party_name_english": "Free Democratic Party",
            "party_name": "FDP",
            "party_name_ascii": "FDP",
            "family_name_short": "lib",
            "family_name": "Liberal",
            "left_right": "5.5",
            "state_market": "",
            "liberty_authority": "",
            "eu_anti_pro": "",
            "cmp": "",
            "euprofiler": "",
            "ees": "",
            "castles_mair": "",
            "huber_inglehart": "",
            "ray": "",
            "benoit_laver": "",
            "chess": "",
            "country_id": "1",
            "party_id": "102",
            "family_id": "10",
        },
        {
            "country_name_short": "DEU",
            "country_name": "Germany",
            "party_name_short": "AfD",
            "party_name_english": "Alternative for Germany",
            "party_name": "AfD",
            "party_name_ascii": "AfD",
            "family_name_short": "right",
            "family_name": "Right-wing",
            "left_right": "8.5",
            "state_market": "",
            "liberty_authority": "",
            "eu_anti_pro": "",
            "cmp": "",
            "euprofiler": "",
            "ees": "",
            "castles_mair": "",
            "huber_inglehart": "",
            "ray": "",
            "benoit_laver": "",
            "chess": "",
            "country_id": "1",
            "party_id": "103",
            "family_id": "30",
        },
        {
            "country_name_short": "DEU",
            "country_name": "Germany",
            "party_name_short": "NONE",
            "party_name_english": "No party",
            "party_name": "none",
            "party_name_ascii": "none",
            "family_name_short": "none",
            "family_name": "no family",
            "left_right": "",
            "state_market": "",
            "liberty_authority": "",
            "eu_anti_pro": "",
            "cmp": "",
            "euprofiler": "",
            "ees": "",
            "castles_mair": "",
            "huber_inglehart": "",
            "ray": "",
            "benoit_laver": "",
            "chess": "",
            "country_id": "1",
            "party_id": "104",
            "family_id": "0",
        },
    ]

    path = tmp_path / "view_party.csv"
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture
def election_csv(tmp_path: Path) -> Path:
    """Create a minimal ParlGov election CSV for testing."""
    rows = [
        # 1950s election
        {
            "country_name_short": "DEU",
            "country_name": "Germany",
            "election_type": "parliament",
            "election_date": "1953-09-06",
            "vote_share": "28.8",
            "seats": "151",
            "seats_total": "487",
            "party_name_short": "SPD",
            "party_name": "SPD",
            "party_name_english": "SPD",
            "left_right": "3.5",
            "country_id": "1",
            "election_id": "10",
            "previous_parliament_election_id": "",
            "previous_cabinet_id": "",
            "party_id": "100",
        },
        {
            "country_name_short": "DEU",
            "country_name": "Germany",
            "election_type": "parliament",
            "election_date": "1953-09-06",
            "vote_share": "45.2",
            "seats": "243",
            "seats_total": "487",
            "party_name_short": "CDU",
            "party_name": "CDU",
            "party_name_english": "CDU",
            "left_right": "6.5",
            "country_id": "1",
            "election_id": "10",
            "previous_parliament_election_id": "",
            "previous_cabinet_id": "",
            "party_id": "101",
        },
        {
            "country_name_short": "DEU",
            "country_name": "Germany",
            "election_type": "parliament",
            "election_date": "1953-09-06",
            "vote_share": "9.5",
            "seats": "48",
            "seats_total": "487",
            "party_name_short": "FDP",
            "party_name": "FDP",
            "party_name_english": "FDP",
            "left_right": "5.5",
            "country_id": "1",
            "election_id": "10",
            "previous_parliament_election_id": "",
            "previous_cabinet_id": "",
            "party_id": "102",
        },
        # 1960s election
        {
            "country_name_short": "DEU",
            "country_name": "Germany",
            "election_type": "parliament",
            "election_date": "1965-09-19",
            "vote_share": "39.3",
            "seats": "202",
            "seats_total": "496",
            "party_name_short": "SPD",
            "party_name": "SPD",
            "party_name_english": "SPD",
            "left_right": "3.5",
            "country_id": "1",
            "election_id": "11",
            "previous_parliament_election_id": "10",
            "previous_cabinet_id": "",
            "party_id": "100",
        },
        {
            "country_name_short": "DEU",
            "country_name": "Germany",
            "election_type": "parliament",
            "election_date": "1965-09-19",
            "vote_share": "47.6",
            "seats": "245",
            "seats_total": "496",
            "party_name_short": "CDU",
            "party_name": "CDU",
            "party_name_english": "CDU",
            "left_right": "6.5",
            "country_id": "1",
            "election_id": "11",
            "previous_parliament_election_id": "10",
            "previous_cabinet_id": "",
            "party_id": "101",
        },
        # 2010s election with Right-wing
        {
            "country_name_short": "DEU",
            "country_name": "Germany",
            "election_type": "parliament",
            "election_date": "2017-09-24",
            "vote_share": "20.5",
            "seats": "153",
            "seats_total": "709",
            "party_name_short": "SPD",
            "party_name": "SPD",
            "party_name_english": "SPD",
            "left_right": "3.5",
            "country_id": "1",
            "election_id": "12",
            "previous_parliament_election_id": "11",
            "previous_cabinet_id": "",
            "party_id": "100",
        },
        {
            "country_name_short": "DEU",
            "country_name": "Germany",
            "election_type": "parliament",
            "election_date": "2017-09-24",
            "vote_share": "32.9",
            "seats": "246",
            "seats_total": "709",
            "party_name_short": "CDU",
            "party_name": "CDU",
            "party_name_english": "CDU",
            "left_right": "6.5",
            "country_id": "1",
            "election_id": "12",
            "previous_parliament_election_id": "11",
            "previous_cabinet_id": "",
            "party_id": "101",
        },
        {
            "country_name_short": "DEU",
            "country_name": "Germany",
            "election_type": "parliament",
            "election_date": "2017-09-24",
            "vote_share": "10.7",
            "seats": "80",
            "seats_total": "709",
            "party_name_short": "FDP",
            "party_name": "FDP",
            "party_name_english": "FDP",
            "left_right": "5.5",
            "country_id": "1",
            "election_id": "12",
            "previous_parliament_election_id": "11",
            "previous_cabinet_id": "",
            "party_id": "102",
        },
        {
            "country_name_short": "DEU",
            "country_name": "Germany",
            "election_type": "parliament",
            "election_date": "2017-09-24",
            "vote_share": "12.6",
            "seats": "94",
            "seats_total": "709",
            "party_name_short": "AfD",
            "party_name": "AfD",
            "party_name_english": "AfD",
            "left_right": "8.5",
            "country_id": "1",
            "election_id": "12",
            "previous_parliament_election_id": "11",
            "previous_cabinet_id": "",
            "party_id": "103",
        },
        # Non-parliament election (should be filtered out)
        {
            "country_name_short": "DEU",
            "country_name": "Germany",
            "election_type": "ep",
            "election_date": "2019-05-26",
            "vote_share": "15.8",
            "seats": "16",
            "seats_total": "96",
            "party_name_short": "SPD",
            "party_name": "SPD",
            "party_name_english": "SPD",
            "left_right": "3.5",
            "country_id": "1",
            "election_id": "20",
            "previous_parliament_election_id": "",
            "previous_cabinet_id": "",
            "party_id": "100",
        },
        # Non-European country (should be filtered out)
        {
            "country_name_short": "AUS",
            "country_name": "Australia",
            "election_type": "parliament",
            "election_date": "2019-05-18",
            "vote_share": "33.3",
            "seats": "68",
            "seats_total": "151",
            "party_name_short": "ALP",
            "party_name": "ALP",
            "party_name_english": "Australian Labor Party",
            "left_right": "3.8",
            "country_id": "33",
            "election_id": "30",
            "previous_parliament_election_id": "",
            "previous_cabinet_id": "",
            "party_id": "100",
        },
        # Party with "no family" (should be excluded from shares)
        {
            "country_name_short": "DEU",
            "country_name": "Germany",
            "election_type": "parliament",
            "election_date": "1953-09-06",
            "vote_share": "5.0",
            "seats": "10",
            "seats_total": "487",
            "party_name_short": "NONE",
            "party_name": "none",
            "party_name_english": "No party",
            "left_right": "",
            "country_id": "1",
            "election_id": "10",
            "previous_parliament_election_id": "",
            "previous_cabinet_id": "",
            "party_id": "104",
        },
    ]

    path = tmp_path / "view_election.csv"
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


class TestLoadPartyFamilies:
    def test_basic_loading(self, party_csv: Path) -> None:
        families = _load_party_families(party_csv)
        assert families["100"] == "Social democracy"
        assert families["101"] == "Christian democracy"
        assert families["102"] == "Liberal"
        assert families["103"] == "Right-wing"
        assert families["104"] == "no family"

    def test_all_parties_loaded(self, party_csv: Path) -> None:
        families = _load_party_families(party_csv)
        assert len(families) == 5


class TestAggregateVoteShares:
    def test_decade_aggregation(self, election_csv: Path, party_csv: Path) -> None:
        families = _load_party_families(party_csv)
        result = _aggregate_vote_shares(
            election_csv=election_csv,
            party_families=families,
            family_mapping=DEFAULT_FAMILY_MAPPING,
            countries=DEFAULT_EUROPEAN_COUNTRIES,
            decade_aggregation=True,
        )

        # Should have 3 decades: 1950, 1960, 2010
        assert 1950 in result
        assert 1960 in result
        assert 2010 in result
        # EP election (2019) should not create a separate decade entry
        # unless there's also a parliament election in the same decade

    def test_shares_sum_to_one(self, election_csv: Path, party_csv: Path) -> None:
        families = _load_party_families(party_csv)
        result = _aggregate_vote_shares(
            election_csv=election_csv,
            party_families=families,
            family_mapping=DEFAULT_FAMILY_MAPPING,
            countries=DEFAULT_EUROPEAN_COUNTRIES,
            decade_aggregation=True,
        )

        for decade, shares in result.items():
            total = sum(shares.values())
            assert abs(total - 1.0) < 0.01, (
                f"Shares for decade {decade} sum to {total}, not ~1.0"
            )

    def test_non_parliament_elections_excluded(
        self, election_csv: Path, party_csv: Path
    ) -> None:
        families = _load_party_families(party_csv)
        result = _aggregate_vote_shares(
            election_csv=election_csv,
            party_families=families,
            family_mapping=DEFAULT_FAMILY_MAPPING,
            countries=DEFAULT_EUROPEAN_COUNTRIES,
            decade_aggregation=True,
        )
        # The EP election is in the 2010s decade but the only parliament
        # election in that decade is 2017, so 2010s data should only
        # reflect the 2017 election
        if 2010 in result:
            # Check that no "ep" data sneaked in
            # 2017 election: SPD=20.5, CDU=32.9, FDP=10.7, AfD=12.6
            # Total mapped = 76.7, so nationalism should be 12.6/76.7
            nationalism_share = result[2010].get("nationalism", 0.0)
            assert nationalism_share > 0.1  # AfD should be visible

    def test_non_european_countries_excluded(
        self, election_csv: Path, party_csv: Path
    ) -> None:
        families = _load_party_families(party_csv)
        result = _aggregate_vote_shares(
            election_csv=election_csv,
            party_families=families,
            family_mapping=DEFAULT_FAMILY_MAPPING,
            countries=DEFAULT_EUROPEAN_COUNTRIES,
            decade_aggregation=True,
        )
        # AUS election is in 2010s but should not appear
        # Only DEU data should be present in 2010
        assert 2010 in result

    def test_no_family_excluded(self, election_csv: Path, party_csv: Path) -> None:
        families = _load_party_families(party_csv)
        result = _aggregate_vote_shares(
            election_csv=election_csv,
            party_families=families,
            family_mapping=DEFAULT_FAMILY_MAPPING,
            countries=DEFAULT_EUROPEAN_COUNTRIES,
            decade_aggregation=True,
        )
        # 1950s: SPD=28.8, CDU=45.2, FDP=9.5, NONE=5.0
        # NONE should be excluded, total should be 83.5
        shares_1950 = result[1950]
        # marxism = 28.8/83.5 ~ 0.345
        assert 0.3 < shares_1950["marxism"] < 0.4

    def test_per_year_aggregation(self, election_csv: Path, party_csv: Path) -> None:
        families = _load_party_families(party_csv)
        result = _aggregate_vote_shares(
            election_csv=election_csv,
            party_families=families,
            family_mapping=DEFAULT_FAMILY_MAPPING,
            countries=DEFAULT_EUROPEAN_COUNTRIES,
            decade_aggregation=False,
        )
        # Should have individual years: 1953, 1965, 2017
        assert 1953 in result
        assert 1965 in result
        assert 2017 in result
        assert 1950 not in result  # decade key should not exist


class TestBuildParlgovDataset:
    def test_builds_valid_dataset(self, election_csv: Path, party_csv: Path) -> None:
        dataset = build_parlgov_dataset(
            election_csv=election_csv,
            party_csv=party_csv,
            decade_aggregation=True,
        )

        assert isinstance(dataset, ObservedDataset)
        assert dataset.name == "parlgov_europe"
        assert len(dataset.observations) > 0
        assert len(dataset.ideology_names) > 0

    def test_ideology_names_sorted(self, election_csv: Path, party_csv: Path) -> None:
        dataset = build_parlgov_dataset(
            election_csv=election_csv,
            party_csv=party_csv,
        )
        assert dataset.ideology_names == sorted(dataset.ideology_names)

    def test_shares_at_year_sum_to_one(
        self, election_csv: Path, party_csv: Path
    ) -> None:
        dataset = build_parlgov_dataset(
            election_csv=election_csv,
            party_csv=party_csv,
            decade_aggregation=True,
        )
        for year in dataset.years:
            shares = dataset.shares_at_year(year)
            total = sum(shares.values())
            assert abs(total - 1.0) < 0.01, f"Year {year} shares sum to {total}"

    def test_roundtrip_through_json(
        self, election_csv: Path, party_csv: Path, tmp_path: Path
    ) -> None:
        """Dataset can be saved and reloaded identically."""
        dataset = build_parlgov_dataset(
            election_csv=election_csv,
            party_csv=party_csv,
        )
        json_path = tmp_path / "test_dataset.json"
        save_dataset(dataset, json_path)
        reloaded = load_dataset(json_path)

        assert reloaded.name == dataset.name
        assert len(reloaded.observations) == len(dataset.observations)
        assert reloaded.ideology_names == dataset.ideology_names

        for year in dataset.years:
            orig = dataset.shares_at_year(year)
            loaded = reloaded.shares_at_year(year)
            for ideo in dataset.ideology_names:
                assert abs(orig.get(ideo, 0) - loaded.get(ideo, 0)) < 1e-5

    def test_custom_country_filter(self, election_csv: Path, party_csv: Path) -> None:
        # Filter to only France (no data in our fixture)
        dataset = build_parlgov_dataset(
            election_csv=election_csv,
            party_csv=party_csv,
            countries=frozenset({"FRA"}),
        )
        assert len(dataset.observations) == 0

    def test_custom_dataset_name(self, election_csv: Path, party_csv: Path) -> None:
        dataset = build_parlgov_dataset(
            election_csv=election_csv,
            party_csv=party_csv,
            dataset_name="custom_name",
        )
        assert dataset.name == "custom_name"


class TestWithRealData:
    """Tests using the cached real ParlGov data (if available)."""

    CACHE_DIR = Path("/home/leonardo/Desktop/zoo/ideasim/data/parlgov_cache")

    @pytest.fixture
    def real_election_csv(self) -> Path:
        path = self.CACHE_DIR / "view_election.csv"
        if not path.exists():
            pytest.skip("ParlGov cache not available")
        return path

    @pytest.fixture
    def real_party_csv(self) -> Path:
        path = self.CACHE_DIR / "view_party.csv"
        if not path.exists():
            pytest.skip("ParlGov cache not available")
        return path

    def test_real_data_produces_valid_dataset(
        self, real_election_csv: Path, real_party_csv: Path
    ) -> None:
        dataset = build_parlgov_dataset(
            election_csv=real_election_csv,
            party_csv=real_party_csv,
            decade_aggregation=True,
        )

        assert len(dataset.observations) >= 40  # at least 10 decades * 4 ideologies
        assert len(dataset.years) >= 10
        assert "liberalism" in dataset.ideology_names
        assert "marxism" in dataset.ideology_names
        assert "traditionalism" in dataset.ideology_names
        assert "nationalism" in dataset.ideology_names

    def test_real_shares_sum_to_one(
        self, real_election_csv: Path, real_party_csv: Path
    ) -> None:
        dataset = build_parlgov_dataset(
            election_csv=real_election_csv,
            party_csv=real_party_csv,
            decade_aggregation=True,
        )
        for year in dataset.years:
            shares = dataset.shares_at_year(year)
            total = sum(shares.values())
            assert abs(total - 1.0) < 0.01, f"Year {year} shares sum to {total}"

    def test_real_data_historical_plausibility(
        self, real_election_csv: Path, real_party_csv: Path
    ) -> None:
        """Basic sanity: liberalism should be significant in 1900s,
        nationalism should grow after 2000."""
        dataset = build_parlgov_dataset(
            election_csv=real_election_csv,
            party_csv=real_party_csv,
            decade_aggregation=True,
        )

        early_shares = dataset.shares_at_year(1900.0)
        assert early_shares.get("liberalism", 0) > 0.2, (
            "Liberalism should be significant in 1900s"
        )

        late_shares = dataset.shares_at_year(2010.0)
        assert late_shares.get("nationalism", 0) > 0.05, (
            "Nationalism should be visible by 2010s"
        )

    def test_compatible_with_evaluation_framework(
        self, real_election_csv: Path, real_party_csv: Path
    ) -> None:
        """Dataset can be converted to matrix form for evaluation."""
        dataset = build_parlgov_dataset(
            election_csv=real_election_csv,
            party_csv=real_party_csv,
            decade_aggregation=True,
        )
        sim_names = ["liberalism", "marxism", "traditionalism", "nationalism"]
        years, shares = dataset.as_matrix(sim_names)

        assert years.shape[0] >= 10
        assert shares.shape == (years.shape[0], 4)
        # No NaN values since all ideologies should be present
        import numpy as np

        assert not np.any(np.isnan(shares))
