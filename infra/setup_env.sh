#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/venv}"
ENV_FILE="${ENV_FILE:-${REPO_ROOT}/.env}"
ENV_TEMPLATE_FILE="${ENV_TEMPLATE_FILE:-${REPO_ROOT}/.env-example}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-${REPO_ROOT}/requirements.txt}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/visdrone}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-${REPO_ROOT}/weights}"
LOGS_ROOT="${LOGS_ROOT:-${REPO_ROOT}/logs}"
NOTEBOOKS_ROOT="${NOTEBOOKS_ROOT:-${REPO_ROOT}/notebooks}"
PYTHON_BIN="${PYTHON_BIN:-}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_PIP_INSTALL="${SKIP_PIP_INSTALL:-0}"
UPGRADE_PIP="${UPGRADE_PIP:-1}"

DEFAULT_ENV_CONTENT=$'NEO4J_URI=bolt://localhost:7687\nNEO4J_USER=neo4j\nNEO4J_PASSWORD=your_password\nOPENAI_API_KEY=\nSEQUENCE_IDS=uav0000009,uav0000013,uav0000073\n'

log() {
  printf '[setup_env] %s\n' "$1"
}

usage() {
  cat <<EOF
Usage: $(basename "$0")

Bootstraps the local development environment by:
  - creating expected repository directories
  - creating a Python virtual environment if missing
  - installing requirements.txt into the virtual environment
  - creating .env from .env-example or built-in defaults if missing

Environment overrides:
  PROJECT_ROOT, VENV_DIR, ENV_FILE, ENV_TEMPLATE_FILE, REQUIREMENTS_FILE
  DATA_ROOT, WEIGHTS_ROOT, LOGS_ROOT, NOTEBOOKS_ROOT
  PYTHON_BIN, DRY_RUN, SKIP_PIP_INSTALL, UPGRADE_PIP
EOF
}

resolve_python_bin() {
  if [ -n "${PYTHON_BIN}" ]; then
    printf '%s\n' "${PYTHON_BIN}"
    return
  fi

  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi

  log "Python 3 is required to set up the environment."
  exit 1
}

run_cmd() {
  if [ "${DRY_RUN}" = "1" ]; then
    log "Would run: $*"
    return
  fi

  "$@"
}

ensure_directories() {
  log "Ensuring repository directories exist"
  run_cmd mkdir -p "${DATA_ROOT}" "${WEIGHTS_ROOT}" "${LOGS_ROOT}" "${NOTEBOOKS_ROOT}"
}

create_virtualenv() {
  if [ -d "${VENV_DIR}" ]; then
    log "Virtual environment already exists at ${VENV_DIR}"
    return
  fi

  log "Creating virtual environment at ${VENV_DIR}"
  run_cmd "$(resolve_python_bin)" -m venv "${VENV_DIR}"
}

install_requirements() {
  local venv_python="${VENV_DIR}/bin/python"

  if [ "${SKIP_PIP_INSTALL}" = "1" ]; then
    log "Skipping dependency installation"
    return
  fi

  if [ ! -f "${REQUIREMENTS_FILE}" ]; then
    log "Requirements file not found at ${REQUIREMENTS_FILE}"
    exit 1
  fi

  if [ ! -x "${venv_python}" ] && [ "${DRY_RUN}" != "1" ]; then
    log "Virtual environment python not found at ${venv_python}"
    exit 1
  fi

  if [ "${UPGRADE_PIP}" = "1" ]; then
    log "Upgrading pip in ${VENV_DIR}"
    run_cmd "${venv_python}" -m pip install --upgrade pip
  fi

  log "Installing dependencies from ${REQUIREMENTS_FILE}"
  run_cmd "${venv_python}" -m pip install -r "${REQUIREMENTS_FILE}"
}

write_default_env_template() {
  if [ "${DRY_RUN}" = "1" ]; then
    log "Would create env template at ${ENV_TEMPLATE_FILE}"
    return
  fi

  printf '%s' "${DEFAULT_ENV_CONTENT}" > "${ENV_TEMPLATE_FILE}"
}

ensure_env_file() {
  if [ -f "${ENV_FILE}" ] && [ -s "${ENV_FILE}" ]; then
    log "Environment file already exists at ${ENV_FILE}"
    return
  fi

  if [ -f "${ENV_TEMPLATE_FILE}" ] && [ -s "${ENV_TEMPLATE_FILE}" ]; then
    log "Creating ${ENV_FILE} from ${ENV_TEMPLATE_FILE}"
    if [ "${DRY_RUN}" = "1" ]; then
      return
    fi
    cp "${ENV_TEMPLATE_FILE}" "${ENV_FILE}"
    return
  fi

  log "Env template missing or empty; using built-in defaults"
  if [ ! -f "${ENV_TEMPLATE_FILE}" ] || [ ! -s "${ENV_TEMPLATE_FILE}" ]; then
    write_default_env_template
  fi

  if [ "${DRY_RUN}" = "1" ]; then
    log "Would create ${ENV_FILE} from default template"
    return
  fi

  cp "${ENV_TEMPLATE_FILE}" "${ENV_FILE}"
}

main() {
  if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    usage
    exit 0
  fi

  ensure_directories
  create_virtualenv
  install_requirements
  ensure_env_file
  log "Environment setup complete."
}

main "$@"
