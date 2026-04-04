#!/usr/bin/env bash
set -e

REPO_URL="https://github.com/godotengine/godot-docs.git"
TARGET_DIR="godot"

echo "Checking Godot docs repo..."

if [ -d "$TARGET_DIR/.git" ]; then
    echo "Repository exists. Updating..."

    cd "$TARGET_DIR"
    git sparse-checkout init --cone 2>/dev/null || true
    git sparse-checkout set classes tutorials getting_started
    git pull

    cd ..

elif [ -d "$TARGET_DIR" ]; then
    echo "Directory exists but is not a git repo. Recreating..."
    rm -rf "$TARGET_DIR"
    git clone --filter=blob:none --sparse "$REPO_URL" "$TARGET_DIR"
    cd "$TARGET_DIR"
    git sparse-checkout set classes tutorials getting_started
    cd ..

else
    echo "Cloning repository..."
    git clone --filter=blob:none --sparse "$REPO_URL" "$TARGET_DIR"
    cd "$TARGET_DIR"
    git sparse-checkout set classes tutorials getting_started
    cd ..
fi

echo "Godot docs ready."
