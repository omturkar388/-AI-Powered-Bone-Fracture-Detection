from pathlib import Path
import hashlib
import argparse

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/raw/FracAtlas"
SPLIT_DIR = DATASET / "Utilities/Fracture Split"
OUTPUT_DIR = ROOT / "data/processed/manifests"

SEED = 42
SPLIT_NAMES = ("train", "valid", "test")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing generated manifests.",
    )
    args = parser.parse_args()

    metadata_path = DATASET / "dataset.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    metadata = pd.read_csv(metadata_path)
    metadata["image_id"] = metadata["image_id"].astype(str).str.strip()

    if metadata["image_id"].duplicated().any():
        raise ValueError("Duplicate image IDs found in dataset.csv.")

    labels = metadata.set_index("image_id")["fractured"].to_dict()
    if not set(labels.values()).issubset({0, 1}):
        raise ValueError("Expected binary labels 0 and 1.")

    # Read and validate the official fractured-image splits.
    official = {}
    for split in SPLIT_NAMES:
        csv_path = SPLIT_DIR / f"{split}.csv"
        df = pd.read_csv(csv_path)

        if "image_id" not in df.columns:
            raise ValueError(f"Missing image_id column: {csv_path}")

        ids = df["image_id"].astype(str).str.strip().tolist()
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate IDs inside {csv_path}")

        for image_id in ids:
            if labels.get(image_id) != 1:
                raise ValueError(
                    f"{image_id} is missing from metadata or not labelled fractured."
                )

            image_path = DATASET / "images/Fractured" / image_id
            if not image_path.is_file():
                raise FileNotFoundError(image_path)

        official[split] = ids

    for i, first in enumerate(SPLIT_NAMES):
        for second in SPLIT_NAMES[i + 1:]:
            overlap = set(official[first]) & set(official[second])
            if overlap:
                raise ValueError(f"Official splits overlap: {first}/{second}")

    all_fractured = {
        p.name for p in (DATASET / "images/Fractured").iterdir()
        if p.is_file()
    }
    official_ids = set().union(*(set(v) for v in official.values()))

    if all_fractured != official_ids:
        missing = sorted(all_fractured - official_ids)
        extra = sorted(official_ids - all_fractured)
        raise ValueError(
            "Official splits do not cover exactly the fractured folder.\n"
            f"Fractured images missing from splits: {missing[:10]}\n"
            f"Split IDs absent from fractured folder: {extra[:10]}"
        )

    # Hash fractured images to detect exact duplicate images in the
    # non-fractured folder, even if their filenames differ.
    fractured_hashes = set()
    for image_id in sorted(all_fractured):
        path = DATASET / "images/Fractured" / image_id
        fractured_hashes.add(sha256_file(path))

    normal_dir = DATASET / "images/Non_fractured"
    normal_files = sorted(
        p for p in normal_dir.iterdir() if p.is_file()
    )

    clean_normals = []
    seen_normal_hashes = set()
    excluded_cross_class = 0
    excluded_normal_duplicates = 0

    for path in normal_files:
        digest = sha256_file(path)

        if digest in fractured_hashes:
            excluded_cross_class += 1
            print(f"Excluding cross-class duplicate: {path.name}")
            continue

        if digest in seen_normal_hashes:
            excluded_normal_duplicates += 1
            print(f"Excluding repeated normal image: {path.name}")
            continue

        seen_normal_hashes.add(digest)
        clean_normals.append(path)

    # Match the official fractured split proportions.
    normal_count = len(clean_normals)
    fracture_counts = np.array(
        [len(official[name]) for name in SPLIT_NAMES], dtype=int
    )
    proportions = fracture_counts / fracture_counts.sum()

    normal_counts = np.floor(normal_count * proportions).astype(int)
    remaining = normal_count - int(normal_counts.sum())
    fractional = normal_count * proportions - normal_counts

    for idx in np.argsort(-fractional)[:remaining]:
        normal_counts[idx] += 1

    rng = np.random.default_rng(SEED)
    shuffled = list(clean_normals)
    rng.shuffle(shuffled)

    normal_splits = {}
    start = 0
    for split, count in zip(SPLIT_NAMES, normal_counts):
        normal_splits[split] = shuffled[start:start + int(count)]
        start += int(count)

    if start != normal_count:
        raise RuntimeError("Normal-image split allocation is incorrect.")

    # Build manifests using project-relative image paths.
    manifests = {}
    for split in SPLIT_NAMES:
        rows = []

        for image_id in official[split]:
            rows.append({
                "image_id": image_id,
                "label": 1,
                "class_name": "fractured",
                "image_path": (
                    Path("data/raw/FracAtlas/images/Fractured") / image_id
                ).as_posix(),
                "split": split,
            })

        for path in normal_splits[split]:
            rows.append({
                "image_id": path.name,
                "label": 0,
                "class_name": "non_fractured",
                "image_path": (
                    Path("data/raw/FracAtlas/images/Non_fractured")
                    / path.name
                ).as_posix(),
                "split": split,
            })

        frame = pd.DataFrame(rows)
        frame = frame.sample(frac=1, random_state=SEED).reset_index(drop=True)
        manifests[split] = frame

    # Verify that no image appears in more than one split.
    for i, first in enumerate(SPLIT_NAMES):
        for second in SPLIT_NAMES[i + 1:]:
            a = set(manifests[first]["image_path"])
            b = set(manifests[second]["image_path"])
            if a & b:
                raise ValueError(f"Manifest overlap: {first}/{second}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    destinations = {
        split: OUTPUT_DIR / f"{split}.csv"
        for split in SPLIT_NAMES
    }

    existing = [str(p) for p in destinations.values() if p.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Manifest files already exist; nothing was overwritten:\n"
            + "\n".join(existing)
            + "\nReview them first, or rerun with --overwrite if intended."
        )

    for split, frame in manifests.items():
        frame.to_csv(destinations[split], index=False)

    print("\n=== DATA PREPARATION COMPLETE ===")
    print("Random seed:", SEED)
    print("Excluded cross-class duplicates:", excluded_cross_class)
    print("Excluded duplicate normal images:", excluded_normal_duplicates)
    print("Fractured images:", len(all_fractured))
    print("Clean non-fractured images:", normal_count)

    for split in SPLIT_NAMES:
        frame = manifests[split]
        counts = frame["class_name"].value_counts().to_dict()
        print(
            f"{split:5} | total={len(frame):4} | "
            f"labels={counts} | saved={destinations[split].relative_to(ROOT)}"
        )

    print("\nOriginal dataset files were not modified.")


if __name__ == "__main__":
    main()
