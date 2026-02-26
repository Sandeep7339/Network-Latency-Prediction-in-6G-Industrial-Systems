#!/usr/bin/env bash
# run_env_check.sh — Create venv, install requirements, run tests.
set -e

VENV_DIR=".venv"

echo "=== 1. Creating virtual environment ==="
if [ ! -d "$VENV_DIR" ]; then
    python -m venv "$VENV_DIR"
    echo "Virtual environment created at $VENV_DIR"
else
    echo "Virtual environment already exists at $VENV_DIR — reusing"
fi

echo ""
echo "=== 2. Activating virtual environment ==="
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate" 2>/dev/null || source "$VENV_DIR/Scripts/activate" 2>/dev/null
echo "Active Python: $(python --version) @ $(which python 2>/dev/null || where python 2>/dev/null)"

echo ""
echo "=== 3. Installing requirements ==="
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo ""
echo "=== 4. Running tests ==="
python -m pytest tests/ -v --tb=short

echo ""
echo "=== Done — environment is ready ==="
