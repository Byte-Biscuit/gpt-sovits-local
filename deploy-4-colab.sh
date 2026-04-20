#!/bin/bash

# --- Colab Deployment & Environment Setup Script ---
# CHECK: Must be executed via 'source' to export variables to current shell
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: Please run as 'source deploy-4-colab.sh' or '. deploy-4-colab.sh'"
    exit 1
fi

echo "--- Initializing GPT-SoVITS Colab Environment ---"

# 1. Define global environment variables for the current shell session
# Set UV to look for or create the virtual environment in /content for speed (local disk)
export UV_PROJECT_ENVIRONMENT="/content/.venv"

# Explicitly set the runtime mode to colab to override any defaults
export ENV_MODE="colab"

# 2. Copy the .env configuration to /content for centralized access
# This ensures config.py loads the correct paths even if the working directory changes
if [ -f "server/.env" ]; then
    cp server/.env /content/.env
    echo "[OK] Copied server/.env to /content/.env"
else
    echo "[WARN] server/.env not found, skipping copy."
fi

# 3. Handle Python Path to ensure local tools are discoverable
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/tools/uvr5:$(pwd)/tools/asr"

# 4. Create symbolic links for pretrained models
# Links Google Drive pretrained models to the local project structure to maintain consistent paths
PRETRAINED_SRC="/content/drive/MyDrive/gpt-sovits/models/pretrained"
PRETRAINED_DST="$(pwd)/GPT_SoVITS/pretrained_models"

if [ -d "$PRETRAINED_SRC" ]; then
    # Remove existing directory if it's not a symlink to prevent conflicts
    if [ -d "$PRETRAINED_DST" ] && [ ! -L "$PRETRAINED_DST" ]; then
        rm -rf "$PRETRAINED_DST"
        echo "[OK] Removed local pretrained_models directory to prepare for symlink."
    fi
    
    # Create the symlink
    if [ ! -L "$PRETRAINED_DST" ]; then
        ln -s "$PRETRAINED_SRC" "$PRETRAINED_DST"
        echo "[OK] Symlink created: $PRETRAINED_DST -> $PRETRAINED_SRC"
    else
        echo "[INFO] Symlink already exists for pretrained_models."
    fi
else
    echo "[WARN] Source pretrained directory not found in Drive: $PRETRAINED_SRC"
fi

echo "--- Environment variables exported ---"
echo "UV_PROJECT_ENVIRONMENT: $UV_PROJECT_ENVIRONMENT"
echo "ENV_MODE: $ENV_MODE"
echo "PYTHONPATH: $PYTHONPATH"
echo ""
echo "Usage: source deploy-4-colab.sh"
echo "Then run: uv sync && uv run server/models_loader.py"
