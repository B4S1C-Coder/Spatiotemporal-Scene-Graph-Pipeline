#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/visdrone}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-${REPO_ROOT}/weights}"
TMP_ROOT="${TMP_ROOT:-${TMPDIR:-/tmp}/visdrone-downloads}"
DRY_RUN="${DRY_RUN:-0}"
PYTHON_BIN="${PYTHON_BIN:-}"

DOWNLOAD_MOT=0
DOWNLOAD_DET=0
DOWNLOAD_WEIGHTS=0

log() {
  printf '[download_visdrone] %s\n' "$1"
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [--mot] [--det] [--weights] [--all]

Downloads project assets into the repository-standard directories:
  data/visdrone
  weights

Defaults to:
  --mot --weights

Options:
  --mot       Download VisDrone MOT train/val archives into data/visdrone
  --det       Download VisDrone DET train/val/test-dev archives into data/visdrone
  --weights   Download yolov8m.pt into weights/
  --all       Download MOT, DET, and weights
  --help      Show this message

Environment overrides:
  DATA_ROOT, WEIGHTS_ROOT, TMP_ROOT, DRY_RUN, PYTHON_BIN
EOF
}

resolve_python_bin() {
  if [ -n "${PYTHON_BIN}" ]; then
    printf '%s\n' "${PYTHON_BIN}"
    return
  fi

  if [ -x "${REPO_ROOT}/venv/bin/python" ]; then
    printf '%s\n' "${REPO_ROOT}/venv/bin/python"
    return
  fi

  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi

  log "Python 3 is required to run this script."
  exit 1
}

run_python() {
  local python_bin
  python_bin="$(resolve_python_bin)"
  "${python_bin}" "$@"
}

ensure_directories() {
  mkdir -p "${DATA_ROOT}" "${WEIGHTS_ROOT}" "${TMP_ROOT}"
}

download_dataset_archive() {
  local dataset_name="$1"
  local file_id="$2"
  local target_dir="${DATA_ROOT}/${dataset_name}"

  if [ -d "${target_dir}" ]; then
    log "Skipping ${dataset_name}; found ${target_dir}"
    return
  fi

  if [ "${DRY_RUN}" = "1" ]; then
    log "Would download ${dataset_name} to ${target_dir}"
    return
  fi

  log "Downloading ${dataset_name} into ${DATA_ROOT}"
  run_python - "${file_id}" "${dataset_name}" "${DATA_ROOT}" "${TMP_ROOT}" <<'PY'
from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

import gdown

file_id, dataset_name, data_root, temp_root = sys.argv[1:]
data_root_path = Path(data_root)
temp_root_path = Path(temp_root)
archive_path = temp_root_path / f"{dataset_name}.zip"
extract_root = data_root_path / dataset_name

temp_root_path.mkdir(parents=True, exist_ok=True)
data_root_path.mkdir(parents=True, exist_ok=True)

url = f"https://drive.google.com/uc?id={file_id}"
gdown.download(url=url, output=str(archive_path), quiet=False, fuzzy=True)

if extract_root.exists():
    shutil.rmtree(extract_root)

with zipfile.ZipFile(archive_path, "r") as archive_file:
    archive_file.extractall(data_root_path)

archive_path.unlink(missing_ok=True)
PY
}

download_yolo_weights() {
  local weight_path="${WEIGHTS_ROOT}/yolov8m.pt"

  if [ -f "${weight_path}" ]; then
    log "Skipping yolov8m.pt; found ${weight_path}"
    return
  fi

  if [ "${DRY_RUN}" = "1" ]; then
    log "Would download yolov8m.pt to ${weight_path}"
    return
  fi

  log "Downloading yolov8m.pt into ${WEIGHTS_ROOT}"
  run_python - "${weight_path}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

from ultralytics.utils.downloads import attempt_download_asset

weight_path = Path(sys.argv[1])
weight_path.parent.mkdir(parents=True, exist_ok=True)
attempt_download_asset(str(weight_path), release="v8.4.0")
PY
}

parse_args() {
  if [ "$#" -eq 0 ]; then
    DOWNLOAD_MOT=1
    DOWNLOAD_WEIGHTS=1
    return
  fi

  while [ "$#" -gt 0 ]; do
    case "$1" in
      --mot)
        DOWNLOAD_MOT=1
        ;;
      --det)
        DOWNLOAD_DET=1
        ;;
      --weights)
        DOWNLOAD_WEIGHTS=1
        ;;
      --all)
        DOWNLOAD_MOT=1
        DOWNLOAD_DET=1
        DOWNLOAD_WEIGHTS=1
        ;;
      --help|-h)
        usage
        exit 0
        ;;
      *)
        log "Unknown option: $1"
        usage
        exit 1
        ;;
    esac
    shift
  done
}

main() {
  parse_args "$@"
  ensure_directories

  if [ "${DOWNLOAD_MOT}" = "1" ]; then
    download_dataset_archive "VisDrone2019-MOT-train" "1-qX2d-P1Xr64ke6nTdlm33om1VxCUTSh"
    download_dataset_archive "VisDrone2019-MOT-val" "1rqnKe9IgU_crMaxRoel9_nuUsMEBBVQu"
  fi

  if [ "${DOWNLOAD_DET}" = "1" ]; then
    download_dataset_archive "VisDrone2019-DET-train" "1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn"
    download_dataset_archive "VisDrone2019-DET-val" "1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59"
    download_dataset_archive "VisDrone2019-DET-test-dev" "1PFdW_VFSCfZ_sTSZAGjQdifF_Xd5mf0V"
  fi

  if [ "${DOWNLOAD_WEIGHTS}" = "1" ]; then
    download_yolo_weights
  fi

  log "Done."
}

main "$@"
