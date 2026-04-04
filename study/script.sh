#!/bin/bash

# Usage: ./script.sh <folder_path> <list.txt>

FOLDER=$1
LIST_FILE=$2

# Check if arguments are provided
if [[ -z "$FOLDER" || -z "$LIST_FILE" ]]; then
    echo "Usage: $0 <folder_path> <list.txt>"
    exit 1
fi

# Find all files in the folder (recursively)
# We use -printf "%f\n" to get only the filename (the title)
find "$FOLDER" -type f -printf "%f\n" | while read -r filename; do
    # Check if the filename is NOT in the list.txt
    if ! grep -qxF "$filename" "$LIST_FILE"; then
        echo "$filename"
    fi
done
