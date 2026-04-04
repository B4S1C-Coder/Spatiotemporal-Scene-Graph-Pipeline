"""Tests for the environment bootstrap shell script."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "infra" / "setup_env.sh"


def run_script(extra_env: dict[str, str] | None = None, *args: str) -> subprocess.CompletedProcess[str]:
    """Run the setup script with a dry-run-friendly environment."""
    environment = os.environ.copy()
    environment.update(
        {
            "DRY_RUN": "1",
            "SKIP_PIP_INSTALL": "1",
        }
    )
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


def test_setup_script_has_valid_bash_syntax() -> None:
    """The setup script should parse successfully."""
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT_PATH)],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert result.returncode == 0, result.stderr


def test_setup_script_reports_expected_default_paths() -> None:
    """Dry-run mode should target the repository venv, data, weights, and env file."""
    result = run_script()

    assert result.returncode == 0, result.stderr
    assert "Would run: mkdir -p" in result.stdout
    assert str(REPO_ROOT / "data" / "visdrone") in result.stdout
    assert str(REPO_ROOT / "weights") in result.stdout
    assert (
        f"Creating virtual environment at {REPO_ROOT / 'venv'}" in result.stdout
        or f"Virtual environment already exists at {REPO_ROOT / 'venv'}" in result.stdout
    )
    assert f"Would run: " in result.stdout
    assert str(REPO_ROOT / ".env") in result.stdout


def test_setup_script_uses_default_env_template_when_template_missing() -> None:
    """Dry-run mode should fall back to built-in env defaults when no template is available."""
    temp_root = REPO_ROOT / "tmp-test-setup-env-missing-template"
    env_file = temp_root / ".env"
    env_template_file = temp_root / ".env-example"

    try:
        result = run_script(
            {
                "PROJECT_ROOT": str(temp_root),
                "ENV_FILE": str(env_file),
                "ENV_TEMPLATE_FILE": str(env_template_file),
                "VENV_DIR": str(temp_root / "venv"),
                "DATA_ROOT": str(temp_root / "data" / "visdrone"),
                "WEIGHTS_ROOT": str(temp_root / "weights"),
                "LOGS_ROOT": str(temp_root / "logs"),
                "NOTEBOOKS_ROOT": str(temp_root / "notebooks"),
            }
        )

        assert result.returncode == 0, result.stderr
        assert "Env template missing or empty; using built-in defaults" in result.stdout
        assert f"Would create env template at {env_template_file}" in result.stdout
        assert f"Would create {env_file} from default template" in result.stdout
    finally:
        if temp_root.exists():
            shutil.rmtree(temp_root)


def test_setup_script_preserves_existing_env_file() -> None:
    """An existing non-empty env file should not be replaced."""
    temp_root = REPO_ROOT / "tmp-test-setup-env-existing-env"
    env_file = temp_root / ".env"
    env_template_file = temp_root / ".env-example"

    try:
        temp_root.mkdir(parents=True, exist_ok=True)
        env_file.write_text("NEO4J_URI=bolt://localhost:7687\n", encoding="utf-8")
        env_template_file.write_text("NEO4J_URI=bolt://localhost:9999\n", encoding="utf-8")

        result = run_script(
            {
                "PROJECT_ROOT": str(temp_root),
                "ENV_FILE": str(env_file),
                "ENV_TEMPLATE_FILE": str(env_template_file),
                "VENV_DIR": str(temp_root / "venv"),
                "DATA_ROOT": str(temp_root / "data" / "visdrone"),
                "WEIGHTS_ROOT": str(temp_root / "weights"),
                "LOGS_ROOT": str(temp_root / "logs"),
                "NOTEBOOKS_ROOT": str(temp_root / "notebooks"),
            }
        )

        assert result.returncode == 0, result.stderr
        assert f"Environment file already exists at {env_file}" in result.stdout
    finally:
        if temp_root.exists():
            shutil.rmtree(temp_root)
