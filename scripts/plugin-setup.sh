#!/usr/bin/env bash
set -euo pipefail

PLUGIN_VERSION="0.3.0"
MODEL_RELEASE_URL="https://github.com/BeadW/synaxi-predict/releases/download/v${PLUGIN_VERSION}/predictor_v1.pkl"

# Resolve user data dir the same way platformdirs does
if [[ "$OSTYPE" == "darwin"* ]]; then
  DATA_DIR="$HOME/Library/Application Support/synaxi-predict"
else
  DATA_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/synaxi-predict"
fi

VERSION_FILE="$DATA_DIR/version"
MODEL_FILE="$DATA_DIR/models/predictor_v1.pkl"

# No-op if already at current version
if [ -f "$VERSION_FILE" ] && [ "$(cat "$VERSION_FILE")" = "$PLUGIN_VERSION" ]; then
  exit 0
fi

echo "synaxi-predict: setting up v${PLUGIN_VERSION}..."

# Install Python package from plugin root
python3 -m pip install -q -e "${CLAUDE_PLUGIN_ROOT}"

# Download model artifact if not already present
mkdir -p "$DATA_DIR/models"
if [ ! -f "$MODEL_FILE" ]; then
  echo "synaxi-predict: downloading model artifact (~190MB)..."
  curl -L --progress-bar "$MODEL_RELEASE_URL" -o "$MODEL_FILE"
fi

echo "$PLUGIN_VERSION" > "$VERSION_FILE"
echo "synaxi-predict: ready"
