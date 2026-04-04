#!/usr/bin/env bash

REPO_URL="https://github.com/godotengine/godot-docs.git"
TARGET_DIR="godot"
TEMP_DIR="godot_clean"

echo "Checking Godot docs repo..."

# Check if folder exists
if [ -d "$TARGET_DIR" ]; then
    # Check if it is a git repo
    if [ -d "$TARGET_DIR/.git" ]; then
        echo "Existing git repository found."
    else
        echo "Folder exists but is not a git repo. Recreating..."
        rm -rf "$TARGET_DIR"
        git clone "$REPO_URL" "$TARGET_DIR"
    fi
else
    echo "Repository not found. Cloning..."
    git clone "$REPO_URL" "$TARGET_DIR"
fi

# Clean previous temp folder
rm -rf "$TEMP_DIR"
mkdir "$TEMP_DIR"

echo "Extracting required documentation..."

mv "$TARGET_DIR/classes" "$TEMP_DIR"
mv "$TARGET_DIR/tutorials" "$TEMP_DIR"
mv "$TARGET_DIR/getting_started" "$TEMP_DIR"

rm -rf "$TARGET_DIR"

mv "$TEMP_DIR" "$TARGET_DIR"

echo "Done."
