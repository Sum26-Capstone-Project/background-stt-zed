#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
VENV_DIR="${VENV_DIR:-${SCRIPT_DIR}/.venv}"
BOOTSTRAP_DIR="${SCRIPT_DIR}/.bootstrap"
SETTINGS_PATH="${SETTINGS_PATH:-${SCRIPT_DIR}/settings.yaml}"
PROJECT_EXTRAS="${PROJECT_EXTRAS:-}"

find_python() {
    local candidates=()

    if [[ -n "${PYTHON:-}" ]]; then
        candidates+=("${PYTHON}")
    fi

    candidates+=(python3.13 python3.12 python3.11 python3.10 python3 python)

    local candidate
    for candidate in "${candidates[@]}"; do
        if command -v "${candidate}" >/dev/null 2>&1; then
            command -v "${candidate}"
            return 0
        fi
    done

    return 1
}

ensure_pip() {
    local python="$1"

    if "${python}" -m pip --version >/dev/null 2>&1; then
        return 0
    fi

    if "${python}" -m ensurepip --upgrade >/dev/null 2>&1; then
        return 0
    fi

    echo "pip is not available in ${VENV_DIR} and ensurepip could not install it." >&2
    return 1
}

create_venv() {
    local python="$1"

    if "${python}" -m venv "${VENV_DIR}"; then
        return 0
    fi

    echo "python -m venv failed; bootstrapping virtualenv with pip"
    mkdir -p "${BOOTSTRAP_DIR}"
    "${python}" -m pip install --upgrade --target "${BOOTSTRAP_DIR}" virtualenv
    PYTHONPATH="${BOOTSTRAP_DIR}${PYTHONPATH:+:${PYTHONPATH}}" "${python}" -m virtualenv "${VENV_DIR}"
}

if [[ -x "${VENV_DIR}/bin/python" ]]; then
    VENV_PYTHON="${VENV_DIR}/bin/python"
else
    BASE_PYTHON="$(find_python || true)"
    if [[ -z "${BASE_PYTHON}" ]]; then
        echo "Could not find a Python 3 interpreter." >&2
        echo "Set PYTHON=/path/to/python3 and run this script again." >&2
        exit 1
    fi

    echo "Creating virtualenv at ${VENV_DIR} using ${BASE_PYTHON}"
    create_venv "${BASE_PYTHON}"
    VENV_PYTHON="${VENV_DIR}/bin/python"
fi

ensure_pip "${VENV_PYTHON}"

if [[ ! -f "${SETTINGS_PATH}" && -f "${SCRIPT_DIR}/settings.example.yaml" ]]; then
    echo "Creating ${SETTINGS_PATH} from settings.example.yaml"
    cp "${SCRIPT_DIR}/settings.example.yaml" "${SETTINGS_PATH}"
fi

if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
    echo "Installing dependencies"
    "${VENV_PYTHON}" -m pip install --upgrade pip setuptools wheel
    # This service only uses torch to run Silero VAD (small, CPU-only model) --
    # faster-whisper itself runs on ctranslate2, not torch. Install the CPU-only
    # wheel explicitly so we don't pull several GB of CUDA deps nobody needs here.
    "${VENV_PYTHON}" -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.0.0"
    if [[ -n "${PROJECT_EXTRAS}" ]]; then
        "${VENV_PYTHON}" -m pip install -e "${SCRIPT_DIR}[${PROJECT_EXTRAS}]"
    else
        "${VENV_PYTHON}" -m pip install -e "${SCRIPT_DIR}"
    fi
fi

export SETTINGS_PATH
echo "Starting Background Realtime STT Service with settings from ${SETTINGS_PATH}"
exec "${VENV_PYTHON}" -m src
