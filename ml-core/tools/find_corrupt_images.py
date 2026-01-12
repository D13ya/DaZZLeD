import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import sys

def check_image(path):
    try:
        with Image.open(path) as img:
            img.convert('RGB').load() # forcing full decode to catch truncation
        return True, None
    except Exception as e:
        return False, str(e)

def main():
    parser = argparse.ArgumentParser(description="Scan a dataset manifest for corrupt images.")
    parser.add_argument('--data-list', required=True, help="Path to the manifest file (e.g., train.txt)")
    args = parser.parse_args()

    data_list_path = Path(args.data_list)
    if not data_list_path.exists():
        print(f"Error: Manifest not found: {data_list_path}")
        sys.exit(1)

    # Read manifest, ignoring blank lines and comments
    lines = [line.strip() for line in data_list_path.read_text().splitlines() if line.strip() and not line.strip().startswith("#")]
    paths = [line.split()[0] for line in lines]
    
    print(f"Scanning {len(paths)} images from {data_list_path}...")
    
    bad_files = []
    
    # Process relative paths if needed
    base_dir = data_list_path.resolve().parent
    
    for i, p_str in enumerate(tqdm(paths)):
        p = Path(p_str)
        if not p.is_absolute():
            p = (base_dir / p).resolve()
            
        is_valid, err = check_image(p)
        if not is_valid:
            print(f"\n[CORRUPT] Line {i+1}: {p}")
            print(f"  Error: {err}")
            bad_files.append((i, p, err))

    print("\n" + "="*50)
    if bad_files:
        print(f"Found {len(bad_files)} corrupt images.")
        print("Recommended action: Remove these lines from your manifest file.")
        for i, p, err in bad_files:
            print(f"  Line {i+1}: {p}")
    else:
        print("All images in the manifest were successfully decoded.")
    print("="*50)

if __name__ == "__main__":
    main()
