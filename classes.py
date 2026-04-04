import os
from pathlib import Path

def process_file(content):
    """
    Dummy function for processing file content.
    Currently returns the content as-is.
    """
    return content

def main():
    input_root = Path("./godot/classes")
    output_root = Path("./godot_chk")

    # Ensure the input directory exists
    if not input_root.exists():
        print(f"Error: Source directory {input_root} not found.")
        return

    for file_path in input_root.rglob("*.rst"):
        # 1. Determine the relative path to maintain structure
        relative_path = file_path.relative_to(input_root)
        
        # 2. Define the new output path with .chk extension
        output_path = output_root / relative_path.with_suffix(".chk")

        # 3. Create the necessary subdirectories in the output folder
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 4. Read, process, and write the file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = f.read()
            
            processed_data = process_file(data)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(processed_data)
            
            print(f"Processed: {relative_path} -> {output_path.name}")
            
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")

if __name__ == "__main__":
    main()
