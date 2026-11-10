#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

CONDA_ENV="${CONDA_ENV:-llm}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11.9}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
MINICONDA_BASE_URL="${MINICONDA_BASE_URL:-https://repo.anaconda.com/miniconda}"
WORKERS="${WORKERS:-4}"
FREEZE_JOBS="${FREEZE_JOBS:-1}"
FREEZE_TASK_WORKERS="${FREEZE_TASK_WORKERS:-1}"
PIPELINE_MODE="${PIPELINE_MODE:-formal}"
PIPELINE_STAGE="${PIPELINE_STAGE:-all}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/pipeline.log}"

find_conda_base() {
  local candidate base
  if [ -n "${CONDA_ROOT:-}" ] && [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
    printf '%s\n' "$CONDA_ROOT"
    return 0
  fi
  for candidate in \
    "$PROJECT_DIR/.conda" \
    "$(dirname "$PROJECT_DIR")/miniconda3" \
    "$HOME/miniconda3"; do
    if [ -f "$candidate/etc/profile.d/conda.sh" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  if command -v conda >/dev/null 2>&1; then
    base="$(conda info --base 2>/dev/null || true)"
    if [ -n "$base" ] && [ -f "$base/etc/profile.d/conda.sh" ]; then
      printf '%s\n' "$base"
      return 0
    fi
  fi
  return 1
}

install_miniconda() {
  local destination architecture installer_name installer_url installer
  destination="${CONDA_ROOT:-$(dirname "$PROJECT_DIR")/miniconda3}"
  architecture="$(uname -m)"
  case "$architecture" in
    x86_64) installer_name="Miniconda3-latest-Linux-x86_64.sh" ;;
    aarch64|arm64) installer_name="Miniconda3-latest-Linux-aarch64.sh" ;;
    *) echo "[ERROR] unsupported architecture: $architecture"; return 1 ;;
  esac
  installer_url="$MINICONDA_BASE_URL/$installer_name"
  installer="$(mktemp "${TMPDIR:-/tmp}/target-imprint-miniconda.XXXXXX.sh")"
  echo "[ENV] installing Miniconda at $destination" >&2
  if command -v curl >/dev/null 2>&1; then
    curl -fL "$installer_url" -o "$installer"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$installer" "$installer_url"
  else
    echo "[ERROR] curl or wget is required to install Miniconda"
    return 1
  fi
  bash "$installer" -b -p "$destination" >&2
  rm -f "$installer"
  printf '%s\n' "$destination"
}

configure_environment() {
  local conda_base requirements_digest requirements_marker
  conda_base="$(find_conda_base || true)"
  if [ -z "$conda_base" ]; then
    conda_base="$(install_miniconda)"
  fi
  # shellcheck disable=SC1090
  source "$conda_base/etc/profile.d/conda.sh"
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true
  if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    echo "[ENV] creating $CONDA_ENV with Python $PYTHON_VERSION"
    conda create -n "$CONDA_ENV" python="$PYTHON_VERSION" -y
  fi
  conda activate "$CONDA_ENV"
  requirements_digest="$(python -c 'import hashlib; print(hashlib.sha256(open("requirements.txt", "rb").read()).hexdigest())')"
  requirements_marker="$CONDA_PREFIX/.target_imprint_requirements_$requirements_digest"
  if [ ! -f "$requirements_marker" ]; then
    echo "[ENV] installing the pinned dependency set"
    python -m pip install --upgrade pip
    python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url "$PYTORCH_INDEX_URL"
    python -m pip install -r requirements.txt
    touch "$requirements_marker"
  fi
}

if [ -z "${HF_HOME:-}" ]; then
  for candidate in \
    "$PROJECT_DIR/hf-cache" \
    "$(dirname "$PROJECT_DIR")/hf-cache" \
    "$PROJECT_DIR/.hf-cache" \
    "$HOME/.cache/huggingface"; do
    if [ -d "$candidate" ]; then
      HF_HOME="$candidate"
      break
    fi
  done
  HF_HOME="${HF_HOME:-$HOME/hf-cache}"
fi
export HF_HOME

CPU_COUNT="${CPU_COUNT:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)}"
if [ "$FREEZE_TASK_WORKERS" -gt 1 ]; then
  THREADS_PER_JOB="${THREADS_PER_JOB:-1}"
else
  THREADS_PER_JOB="${THREADS_PER_JOB:-$((CPU_COUNT / FREEZE_JOBS))}"
fi
if [ "$THREADS_PER_JOB" -lt 1 ]; then THREADS_PER_JOB=1; fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$THREADS_PER_JOB}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$THREADS_PER_JOB}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$THREADS_PER_JOB}"

mkdir -p "$LOG_DIR"
: > "$LOG_FILE"

main() {
  echo "=== pipeline start $(date -Is) ==="
  echo "host: $(hostname)"
  echo "cwd: $PROJECT_DIR"
  echo "workers: $WORKERS"
  echo "freeze_jobs: $FREEZE_JOBS"
  echo "freeze_task_workers: $FREEZE_TASK_WORKERS"
  echo "mode: $PIPELINE_MODE"
  echo "stage: $PIPELINE_STAGE"
  echo "hf_home: $HF_HOME"

  configure_environment
  echo "python: $(python --version 2>&1)"
  echo "conda_env: $CONDA_ENV"

  python runner.py \
    --stage "$PIPELINE_STAGE" \
    --mode "$PIPELINE_MODE" \
    --workers "$WORKERS" \
    --freeze-jobs "$FREEZE_JOBS" \
    --freeze-task-workers "$FREEZE_TASK_WORKERS" \
    "$@"

  echo "=== pipeline done $(date -Is) ==="
}

main "$@" 2>&1 | tee -a "$LOG_FILE"
