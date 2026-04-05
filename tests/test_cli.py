"""Tests for CLI commands."""

from __future__ import annotations

import json

import pytest

from ideasim.cli import main


class TestValidateCommand:
    def test_validate_default_config(self) -> None:
        """validate with no args should pass on default config."""
        # Should not raise or sys.exit
        main(["validate"])

    def test_validate_bad_config(self, tmp_path: object) -> None:
        """validate should exit 1 on an invalid config."""
        import pathlib

        path = pathlib.Path(str(tmp_path)) / "bad.json"
        # Write a config with wrong-shaped matrices
        from ideasim.defaults import make_default_config
        from ideasim.io import save_config

        config = make_default_config()
        save_config(config, path)

        # Now corrupt the file: change initial_shares
        with open(path) as f:
            data = json.load(f)
        data["initial_shares"] = [0.5, 0.5]  # wrong length
        with open(path, "w") as f:
            json.dump(data, f)

        with pytest.raises(SystemExit) as exc_info:
            main(["validate", str(path)])
        assert exc_info.value.code == 1


class TestCheckCommand:
    def test_check_default_json(self) -> None:
        """check --json should produce valid JSON output."""
        import io
        import sys

        old_stdout = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            main(["check", "--json"])
        finally:
            sys.stdout = old_stdout
        output = buf.getvalue()
        # Find the JSON part (after "Running simulation" etc.)
        lines = output.strip().split("\n")
        # The JSON block starts with a line beginning with '{'
        json_start = next(
            i for i, line in enumerate(lines) if line.strip().startswith("{")
        )
        json_text = "\n".join(lines[json_start:])
        data = json.loads(json_text)
        assert data["all_passed"] is True
        assert "checks" in data


class TestSensitivityCommand:
    def test_sensitivity_oat_only(self, tmp_path: object) -> None:
        """sensitivity --no-lhs should produce results with OAT data only."""
        import pathlib

        out = pathlib.Path(str(tmp_path)) / "sens.json"
        main(["sensitivity", "-o", str(out), "--oat-steps", "3", "--no-lhs"])
        assert out.exists()
        with open(out) as f:
            data = json.load(f)
        assert len(data["oat"]) > 0
        assert len(data["lhs_runs"]) == 0

    def test_sensitivity_lhs_only(self, tmp_path: object) -> None:
        """sensitivity --no-oat should produce results with LHS data only."""
        import pathlib

        out = pathlib.Path(str(tmp_path)) / "sens.json"
        main(["sensitivity", "-o", str(out), "--lhs-samples", "3", "--no-oat"])
        assert out.exists()
        with open(out) as f:
            data = json.load(f)
        assert len(data["oat"]) == 0
        assert len(data["lhs_runs"]) == 3
        assert len(data["lhs_stats"]) > 0

    def test_sensitivity_json_output(self, tmp_path: object) -> None:
        """sensitivity --json should produce a valid JSON output file."""
        import pathlib

        out = pathlib.Path(str(tmp_path)) / "sens.json"
        main(
            [
                "sensitivity",
                "--json",
                "--oat-steps",
                "2",
                "--lhs-samples",
                "2",
                "-o",
                str(out),
            ]
        )
        assert out.exists()
        with open(out) as f:
            data = json.load(f)
        assert "oat" in data
        assert "lhs_stats" in data

    def test_sensitivity_both_disabled_exits(self) -> None:
        """sensitivity --no-oat --no-lhs should exit with error."""
        with pytest.raises(SystemExit) as exc_info:
            main(["sensitivity", "--no-oat", "--no-lhs"])
        assert exc_info.value.code == 1


class TestSimulateCommand:
    def test_simulate_default(self, tmp_path: object) -> None:
        """simulate with default config should produce results file."""
        import pathlib

        out = pathlib.Path(str(tmp_path)) / "results.json"
        main(["simulate", "-o", str(out)])
        assert out.exists()
        with open(out) as f:
            data = json.load(f)
        assert "history" in data
        assert "summary" in data
