import os
import sys

def find_files_by_content(target_folder, pattern_file):
    # 1. Read the exact multi-line pattern from your reference file
    try:
        with open(pattern_file, 'r', encoding='utf-8') as f:
            search_pattern = f.read()
    except FileNotFoundError:
        print(f"Error: Pattern file '{pattern_file}' not found.")
        return

    # 2. Recursively walk through the directory
    for root, dirs, files in os.walk(target_folder):
        for filename in files:
            file_path = os.path.join(root, filename)

            try:
                # Open each file and check for the pattern
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if search_pattern in content:
                        # Print only the filename (the "title")
                        print(filename)
            except Exception:
                # This skips files that can't be read (like binaries or locked files)
                continue

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <folder_path> <pattern_file.txt>")
    else:
        find_files_by_content(sys.argv[1], sys.argv[2])
