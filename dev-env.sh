#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${ROOT_DIR}/env"
PY_CHECK='import numpy,pandas,sklearn,scipy,matplotlib,seaborn,joblib,pyDOE2,yaml; print("ok")'

is_sourced() {
  [[ "${BASH_SOURCE[0]}" != "${0}" ]]
}

status() {
  local py_path pip_path active
  py_path="$(command -v python || true)"
  pip_path="$(command -v pip || true)"
  if [[ -n "${VIRTUAL_ENV:-}" && "${VIRTUAL_ENV}" == "${ENV_DIR}" ]]; then
    active="yes (project env)"
  elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    active="yes (different env: ${VIRTUAL_ENV})"
  else
    active="no"
  fi

  echo "repo:          ${ROOT_DIR}"
  echo "expected env:  ${ENV_DIR}"
  echo "active env:    ${active}"
  echo "python:        ${py_path:-not found}"
  echo "pip:           ${pip_path:-not found}"
  python -c 'import sys; print(f"python exe:    {sys.executable}")' 2>/dev/null || true
  python -c 'import sys; print(f"python ver:    {sys.version.split()[0]}")' 2>/dev/null || true
}

enter_env() {
  if [[ ! -d "${ENV_DIR}" ]]; then
    echo "env directory not found at ${ENV_DIR}" >&2
    echo "run: ./dev-env.sh ensure" >&2
    return 1
  fi

  if is_sourced; then
    # shellcheck disable=SC1091
    source "${ENV_DIR}/bin/activate"
    echo "activated: ${ENV_DIR}"
  else
    echo "this command must be sourced to persist activation:" >&2
    echo "  source ./dev-env.sh enter" >&2
    return 1
  fi
}

ensure_env() {
  if [[ ! -d "${ENV_DIR}" ]]; then
    if python3 -m venv "${ENV_DIR}" >/dev/null 2>&1; then
      echo "created env with venv"
    else
      echo "venv unavailable, trying virtualenv fallback"
      python3 -m virtualenv \
        --app-data /tmp/virtualenv-appdata \
        --no-periodic-update \
        --never-download \
        "${ENV_DIR}"
      echo "created env with virtualenv"
    fi
  else
    echo "env already exists"
  fi

  # shellcheck disable=SC1091
  source "${ENV_DIR}/bin/activate"
  pip install -r "${ROOT_DIR}/requirements.txt"
}

doctor() {
  python -c "${PY_CHECK}"
}

usage() {
  cat <<'EOF'
Usage:
  ./dev-env.sh status    # show which python/pip/env is active
  source ./dev-env.sh enter
  ./dev-env.sh ensure    # create env (venv or virtualenv fallback) and install deps
  ./dev-env.sh doctor    # verify core imports
EOF
}

cmd="${1:-status}"
case "${cmd}" in
  status) status ;;
  enter) enter_env ;;
  ensure) ensure_env ;;
  doctor) doctor ;;
  *) usage; exit 1 ;;
esac
