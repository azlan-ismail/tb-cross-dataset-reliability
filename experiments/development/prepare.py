"""Download NLM datasets, audit duplicates, and save immutable study manifests.

No patient images leave this machine. This module does not execute input scripts.
"""
import argparse
import csv
import hashlib
import json
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedGroupKFold

SOURCES = {
    "montgomery": "https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip",
    "shenzhen": "https://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip",
}
REVISION = "110cbc18d5133582e320b43d53bf5c44e410c936"


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or list(rows[0])
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def download(root):
    root = Path(root)
    archive_dir = root / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for source, url in SOURCES.items():
        archive = archive_dir / f"{source}.zip"
        if not archive.exists():
            print(f"Downloading {source} (large archive; Shenzhen about 3.8 GB)", flush=True)
            partial = archive.with_suffix(".partial")
            urllib.request.urlretrieve(url, partial)
            with zipfile.ZipFile(partial) as z:
                bad = z.testzip()
                if bad:
                    raise ValueError(f"Corrupt archive member: {bad}")
            partial.replace(archive)
        target = (root / source).resolve()
        target.mkdir(parents=True, exist_ok=True)
        marker = target / "extraction.json"
        archive_hash = digest(archive)
        if marker.exists():
            if json.loads(marker.read_text())["archive_sha256"] != archive_hash:
                raise ValueError("Archive changed; use a new data directory")
            continue
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                dest = (target / member.filename).resolve()
                if not dest.is_relative_to(target):
                    raise ValueError("Unsafe archive path")
                # Extract only image and lung-mask PNGs; never execute archive content.
                if member.filename.lower().endswith(".png"):
                    z.extract(member, target)
        write_json(marker, {"source_url": url, "archive_sha256": archive_hash})
        print(f"Extracted {source}", flush=True)


def image_signature(path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("L")
        size = im.size
        pixels = np.asarray(im)
        pixel_hash = hashlib.sha256(str(size).encode() + pixels.tobytes()).hexdigest()
        thumbnail = np.asarray(im.resize((9, 8), Image.Resampling.LANCZOS))
        bits = thumbnail[:, 1:] > thumbnail[:, :-1]
        dhash = int.from_bytes(np.packbits(bits.ravel()).tobytes(), "big")
    return size, pixel_hash, f"{dhash:016x}"


def inventory(root):
    root = Path(root).resolve()
    rows = []
    for source in SOURCES:
        for path in sorted((root / source).rglob("*.png")):
            if path.parent.name != "CXR_png":
                continue
            match = re.fullmatch(r"(?:MCUCXR|CHNCXR)_(\d+)_([01])", path.stem)
            if not match:
                raise ValueError(f"Unrecognized label convention: {path.name}")
            size, pixel_hash, dhash = image_signature(path)
            rows.append({"image_id": source + ":" + path.name,
                         "patient_id": source + ":" + match[1],
                         "source": source, "label": int(match[2]),
                         "path": path.relative_to(root).as_posix(),
                         "sha256": digest(path), "pixel_sha256": pixel_hash,
                         "dhash": dhash, "width": size[0], "height": size[1]})
    for source in SOURCES:
        subset = [r for r in rows if r["source"] == source]
        if not subset or set(r["label"] for r in subset) != {0, 1}:
            raise ValueError(f"Missing images or class in {source}")
    return rows


def duplicate_groups(rows, max_distance=4):
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    candidates = []
    for i, a in enumerate(rows):
        for j in range(i):
            b = rows[j]
            distance = (int(a["dhash"], 16) ^ int(b["dhash"], 16)).bit_count()
            exact = a["sha256"] == b["sha256"] or a["pixel_sha256"] == b["pixel_sha256"]
            same_patient = a["patient_id"] == b["patient_id"]
            near = distance <= max_distance
            if exact or same_patient or near:
                if exact and a["label"] != b["label"]:
                    raise ValueError(f"Duplicate image with conflicting labels: {a['image_id']}")
                if exact or same_patient:
                    parent[find(i)] = find(j)
                candidates.append({"image_a": a["image_id"], "image_b": b["image_id"],
                                   "exact": exact, "same_patient": same_patient,
                                   "dhash_distance": distance,
                                   "cross_source": a["source"] != b["source"]})
    members = {}
    for i, row in enumerate(rows):
        members.setdefault(find(i), []).append(row["image_id"])
    for i, row in enumerate(rows):
        row["group_id"] = hashlib.sha256("|".join(sorted(members[find(i)])).encode()).hexdigest()[:16]
    return candidates


def make_split(rows, test_source="montgomery", seed=42):
    # Hold the entire target source out. Conservatively remove source-side members
    # of suspected cross-source duplicate groups before splitting development data.
    test_groups = {r["group_id"] for r in rows if r["source"] == test_source}
    development = [r for r in rows if r["source"] != test_source and r["group_id"] not in test_groups]
    y = np.array([r["label"] for r in development])
    groups = np.array([r["group_id"] for r in development])
    if min(sum(y == 0), sum(y == 1)) < 5:
        raise ValueError("Insufficient development images for five-fold stratification")
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train, val = next(splitter.split(np.zeros(len(y)), y, groups))
    assignment = {development[i]["image_id"]: "train" for i in train}
    assignment.update({development[i]["image_id"]: "validation" for i in val})
    result = []
    for row in rows:
        partition = "test" if row["source"] == test_source else assignment.get(row["image_id"], "excluded_overlap")
        result.append({**row, "split": partition})
    validate_split(result)
    return result


def validate_split(rows):
    groups = {}
    for row in rows:
        if row["split"] == "excluded_overlap":
            continue
        groups.setdefault(row["group_id"], set()).add(row["split"])
    if any(len(s) > 1 for s in groups.values()):
        raise ValueError("A patient/duplicate group crosses partitions")
    for name in ("train", "validation", "test"):
        if {int(r["label"]) for r in rows if r["split"] == name} != {0, 1}:
            raise ValueError(f"Partition {name} must have both classes")


def prepare(root, output, test_source="montgomery", seed=42):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new study directory; existing manifests are never overwritten")
    rows = inventory(root)
    candidates = duplicate_groups(rows)
    decisions_path = Path(__file__).with_name("adjudicated_pairs.csv")
    decisions = read_csv(decisions_path)
    expected = {(p["image_a"], p["image_b"]) for p in candidates if p["cross_source"]}
    reviewed = {(p["image_a"], p["image_b"]) for p in decisions}
    if expected != reviewed or len(decisions) != len(reviewed):
        raise ValueError("Cross-source review does not match current image candidates")
    if any(p["decision"] != "distinct" or not p["reason"] or not p["reviewer"] for p in decisions):
        raise ValueError("This reviewed protocol supports distinct pairs only; resolve any duplicate/uncertain decisions explicitly")
    split = make_split(rows, test_source, seed)
    output.mkdir(parents=True)
    write_csv(output / "adjudicated_pairs.csv", decisions)
    write_csv(output / "inventory.csv", rows)
    write_csv(output / "duplicate_candidates.csv", candidates,
              ["image_a", "image_b", "exact", "same_patient", "dhash_distance", "cross_source"])
    write_csv(output / "manifest.csv", split)
    counts = {s: {str(y): sum(r["split"] == s and r["label"] == y for r in split)
                  for y in (0, 1)} for s in ("train", "validation", "test", "excluded_overlap")}
    write_json(output / "protocol.json", {
        "protocol": "PV-classifier-v2-exploratory",
        "review_complete": True,
        "review_sha256": digest(output / "adjudicated_pairs.csv"),
        "review_scope": "AI contact-sheet review of 35 cross-source candidates; all distinct image content. No clinical or patient-identity adjudication. Other dHash candidates recorded but not visually adjudicated.", "test_source": test_source, "split_seed": seed,
        "model": "microsoft/rad-dino", "revision": REVISION,
        "manifest_sha256": digest(output / "manifest.csv"), "counts": counts,
        "near_duplicate_policy": "dHash candidates recorded only; exact pixels and same patient grouped. Visual adjudication required before training.",
        "grouping": "source-qualified filename patient identifier plus exact duplicate components",
        "limitations": ["No guarantee of detecting all transformed duplicates",
                        "Patient identity inferred from dataset filenames; verify metadata",
                        "Cross-source downstream holdout does not prove absence of foundation-model exposure"],
    })
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--study", default="study")
    p.add_argument("--download", action="store_true")
    p.add_argument("--test-source", choices=list(SOURCES), default="montgomery")
    a = p.parse_args()
    if a.download:
        download(a.data)
    prepare(a.data, a.study, a.test_source)
