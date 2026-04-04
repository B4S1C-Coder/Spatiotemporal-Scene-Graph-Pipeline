"""Tests for the VisDrone download helper script."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "infra" / "download_visdrone.sh"


def run_script(*args: str, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run the download script and return the completed process."""
    environment = os.environ.copy()
    environment["DRY_RUN"] = "1"
    if extra_env:
        environment.update(extra_env)

    return subprocess.run(
        ["bash", str(SCRIPT_PATH), *args],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=environment,
    )


def test_download_script_has_valid_bash_syntax() -> None:
    """The shell script should parse successfully."""
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT_PATH)],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert result.returncode == 0, result.stderr


def test_download_script_defaults_to_mot_and_weights_directories() -> None:
    """Dry-run mode should target the repository data and weights directories."""
    result = run_script()

    assert result.returncode == 0, result.stderr
    assert "Would download VisDrone2019-MOT-train to" in result.stdout
    assert str(REPO_ROOT / "data" / "visdrone" / "VisDrone2019-MOT-train") in result.stdout
    assert str(REPO_ROOT / "data" / "visdrone" / "VisDrone2019-MOT-val") in result.stdout
    assert str(REPO_ROOT / "weights" / "yolov8m.pt") in result.stdout


def test_download_script_all_option_includes_det_archives() -> None:
    """The --all flag should include DET assets in the data directory."""
    result = run_script("--all")

    assert result.returncode == 0, result.stderr
    assert str(REPO_ROOT / "data" / "visdrone" / "VisDrone2019-DET-train") in result.stdout
    assert str(REPO_ROOT / "data" / "visdrone" / "VisDrone2019-DET-val") in result.stdout
    assert str(REPO_ROOT / "data" / "visdrone" / "VisDrone2019-DET-test-dev") in result.stdout


def test_download_script_rejects_unknown_flags() -> None:
    """Unknown flags should fail fast with a usage message."""
    result = run_script("--unknown")

    assert result.returncode != 0
    assert "Unknown option: --unknown" in result.stdout
    assert "Usage:" in result.stdout
