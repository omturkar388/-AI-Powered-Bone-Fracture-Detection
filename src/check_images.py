from pathlib import Path
import shutil
import warnings

import pandas as pd
from PIL import Image, ImageFile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "data/processed/manifests"

# Reject incomplete images instead of silently accepting truncated JPEGs.
ImageFile.LOAD_TRUNCATED_IMAGES = False

path_columns = ["image_path", "filepath", "file_path", "path", "image"]
bad_paths = set()
checked_paths = set()

def get_image_path(row, column):
    return Path(str(row[column])).expanduser()

def is_valid_image(path):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with Image.open(path) as img:
                img.verify()

            # Reopen and fully decode the image.
            with Image.open(path) as img:
                img.load()

        return True
    except Exception as exc:
        print(f"BAD IMAGE: {path} | {type(exc).__name__}: {exc}")
        return False

manifests = sorted(MANIFEST_DIR.glob("*.csv"))
if not manifests:
    raise SystemExit(f"No manifest CSV files found in {MANIFEST_DIR}")

for manifest in manifests:
    df = pd.read_csv(manifest)

    column = next((c for c in path_columns if c in df.columns), None)
    if column is None:
        raise SystemExit(
            f"Could not identify image path column in {manifest.name}. "
            f"Columns: {list(df.columns)}"
        )

    print(f"\nChecking {manifest.name}: {len(df)} rows")

    for value in df[column].dropna():
        path = get_image_path({column: value}, column)
        key = str(path)

        if key in checked_paths:
            continue

        checked_paths.add(key)

        if not path.is_file() or not is_valid_image(path):
            bad_paths.add(key)

print(f"\nUnique image paths checked: {len(checked_paths)}")
print(f"Invalid or missing image paths: {len(bad_paths)}")

if not bad_paths:
    print("All manifest images passed validation. No files changed.")
    raise SystemExit(0)

for manifest in manifests:
    df = pd.read_csv(manifest)
    column = next(c for c in path_columns if c in df.columns)

    original_count = len(df)
    row_paths = df[column].astype(str).map(
        lambda p: str(Path(p).expanduser())
    )
    cleaned = df.loc[~row_paths.isin(bad_paths)].copy()
    removed = original_count - len(cleaned)

    if removed:
        backup = manifest.with_suffix(manifest.suffix + ".bak")
        if not backup.exists():
            shutil.copy2(manifest, backup)

        cleaned.to_csv(manifest, index=False)
        print(
            f"{manifest.name}: removed {removed} invalid row(s); "
            f"remaining {len(cleaned)}. Backup: {backup.name}"
        )

print("\nManifest cleanup complete. Original dataset images were not deleted.")
