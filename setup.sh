#!/usr/bin/env bash
set -euo pipefail

# setup.sh - create venv, install requirements, create placeholders for empty folders
# Usage: ./setup.sh

PYTHON_BIN=${PYTHON_BIN:-python3}
VENV_DIR=${VENV_DIR:-venv}

echo "Using Python: $(command -v $PYTHON_BIN || true)"

# create virtual environment
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment in ./$VENV_DIR ..."
  $PYTHON_BIN -m venv "$VENV_DIR"
fi

# activate environment for this shell session (informational)
echo "To activate the virtualenv in this shell run:"
echo "  source $VENV_DIR/bin/activate"

# install pip upgrades and requirements
echo "Upgrading pip..."
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel

if [ -f requirements.txt ]; then
  echo "Installing requirements from requirements.txt ..."
  "$VENV_DIR/bin/pip" install -r requirements.txt
else
  echo "No requirements.txt found. Please add one to continue."
fi

# Create recommended folder structure and .keep files
echo "Creating repo folders (placeholders only)..."
mkdir -p Tensorflow/workspace/images/collected_images
mkdir -p Tensorflow/workspace/images/model
touch Tensorflow/workspace/images/collected_images/.keep
touch Tensorflow/workspace/images/model/.keep

echo "Setup done. Activate virtualenv with: source $VENV_DIR/bin/activate"
