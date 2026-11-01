"""Frozen RAD-DINO features, two classifier heads, locked external evaluation.

Commands: encode, fit, evaluate. No image-based clinical claims are made by this code.
"""
import argparse
import importlib.metadata
import json
import os
import platform
import random
from pathlib import Path

import numpy as np
from prepare import digest, read_csv, write_csv, write_json, validate_split, REVISION


def metadata():
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("torch", "transformers", "numpy", "scikit-learn", "pillow", "huggingface-hub"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not installed"
    result["experiment_code_sha256"] = digest(__file__)
    result["prepare_code_sha256"] = digest(Path(__file__).with_name("prepare.py"))
    return result


def locked_manifest(study):
    study = Path(study)
    protocol = json.loads((study / "protocol.json").read_text())
    if digest(study / "manifest.csv") != protocol["manifest_sha256"]:
        raise ValueError("Study manifest changed after protocol was created")
    if protocol.get("protocol") == "PV-classifier-v2-exploratory":
        if not protocol.get("review_complete") or digest(study / "adjudicated_pairs.csv") != protocol.get("review_sha256"):
            raise ValueError("Visual review record missing or changed")
    rows = read_csv(study / "manifest.csv")
    validate_split(rows)
    return rows, protocol


def seed_everything(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(min(4, os.cpu_count() or 1))


def head_model(dim):
    import torch.nn as nn
    return nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 256), nn.GELU(),
                         nn.Dropout(0.3), nn.Linear(256, 2))


def encode(data, study, batch_size=4, allow_cpu=False):
    import torch
    from PIL import Image, ImageOps
    from transformers import AutoImageProcessor, AutoModel
    rows, protocol = locked_manifest(study)
    if not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError("GPU required for full feature extraction; use Colab GPU or explicitly --allow-cpu")
    seed_everything(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(study) / "features"
    root.mkdir(exist_ok=True)
    cache_meta = {"manifest_sha256": protocol["manifest_sha256"],
                  "model": protocol["model"], "revision": REVISION,
                  "preprocessing": "AutoImageProcessor(use_fast=False), RGB, EXIF transpose, no augmentation",
                  "device": device, "environment": metadata()}
    old_meta = root / "metadata.json"
    if old_meta.exists():
        old = json.loads(old_meta.read_text())
        if any(old[k] != cache_meta[k] for k in ("manifest_sha256", "model", "revision", "preprocessing", "environment", "device")):
            raise ValueError("Feature cache configuration changed. Use a new study directory")
    write_json(old_meta, cache_meta)
    processor = AutoImageProcessor.from_pretrained(protocol["model"], revision=REVISION, use_fast=False)
    model = AutoModel.from_pretrained(protocol["model"], revision=REVISION, use_safetensors=True)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    processor.save_pretrained(root / "processor")
    pending = []
    for row in rows:
        if row["split"] == "excluded_overlap":
            continue
        path = Path(data) / row["path"]
        if digest(path) != row["sha256"]:
            raise ValueError(f"Image content changed: {row['image_id']}")
        dest = root / f"{row['sha256']}.npy"
        if not dest.exists():
            pending.append(row)
    with torch.inference_mode():
        for i in range(0, len(pending), batch_size):
            batch = pending[i:i + batch_size]
            images = []
            for row in batch:
                with Image.open(Path(data) / row["path"]) as im:
                    images.append(ImageOps.exif_transpose(im).convert("RGB"))
            inputs = processor(images=images, return_tensors="pt").to(device)
            vectors = model(**inputs).last_hidden_state[:, 0, :].float().cpu().numpy()
            for row, vector in zip(batch, vectors):
                dest = root / f"{row['sha256']}.npy"
                partial = dest.with_suffix(".partial")
                with open(partial, "wb") as f:
                    np.save(f, vector, allow_pickle=False)
                partial.replace(dest)
            print(f"Encoded {min(i + len(batch), len(pending))}/{len(pending)} pending images", flush=True)
    print("Feature extraction complete", flush=True)


def load_features(study, rows, split):
    selected = [r for r in rows if r["split"] == split]
    x = np.stack([np.load(Path(study) / "features" / f"{r['sha256']}.npy", allow_pickle=False) for r in selected])
    y = np.array([int(r["label"]) for r in selected])
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Invalid feature cache")
    return x.astype(np.float32), y, selected


def select_threshold(y, p, target=0.90):
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y, p, drop_intermediate=False)
    eligible = np.where((tpr >= target) & np.isfinite(thresholds))[0]
    if len(eligible) == 0:
        raise ValueError("No finite threshold reaches target validation sensitivity")
    # Lowest false-positive rate, then highest threshold among ties.
    idx = min(eligible, key=lambda k: (fpr[k], -thresholds[k]))
    return float(thresholds[idx])


def metric_values(y, p, threshold):
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y, p >= threshold, labels=[0, 1]).ravel()
    return {"auroc": float(roc_auc_score(y, p)),
            "average_precision": float(average_precision_score(y, p)),
            "sensitivity": float(tp / (tp + fn)), "specificity": float(tn / (tn + fp)),
            "ppv": float(tp / (tp + fp)) if tp + fp else None,
            "npv": float(tn / (tn + fn)) if tn + fn else None,
            "brier": float(brier_score_loss(y, p)),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}


def bootstrap(y, p, groups, threshold, repeats=2000, seed=2026):
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    indices = {g: np.where(groups == g)[0] for g in unique}
    samples = {k: [] for k in ("auroc", "average_precision", "sensitivity", "specificity", "brier")}
    for _ in range(repeats):
        ix = np.concatenate([indices[g] for g in rng.choice(unique, len(unique), replace=True)])
        if len(np.unique(y[ix])) < 2:
            continue
        vals = metric_values(y[ix], p[ix], threshold)
        for k in samples:
            samples[k].append(vals[k])
    if len(samples["auroc"]) < 0.9 * repeats:
        raise ValueError("Too few valid bootstrap samples; dataset is unsuitable for this interval method")
    return {k: [float(v) for v in np.percentile(values, [2.5, 97.5])] for k, values in samples.items()}


def feature_fingerprint(study, rows):
    import hashlib
    h = hashlib.sha256((Path(study) / "features" / "metadata.json").read_bytes())
    for row in rows:
        if row["split"] != "excluded_overlap":
            path = Path(study) / "features" / f"{row['sha256']}.npy"
            h.update((row["image_id"] + digest(path)).encode())
    return h.hexdigest()


def fit(study, seeds=(42, 43, 44), epochs=20):
    import torch
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    rows, protocol = locked_manifest(study)
    out = Path(study) / "models"
    if out.exists():
        raise FileExistsError("Models already exist. Do not overwrite a locked study")
    xtrain, ytrain, _ = load_features(study, rows, "train")
    xval, yval, valrows = load_features(study, rows, "validation")
    cache_hash = feature_fingerprint(study, rows)
    out.mkdir()
    config = {"manifest_sha256": protocol["manifest_sha256"], "feature_fingerprint": cache_hash,
              "environment": metadata(), "seeds": list(seeds), "epochs": epochs,
              "optimizer": "AdamW", "lr": 1e-4, "weight_decay": 1e-4, "batch_size": 8,
              "selection": "highest validation AUROC; earliest epoch for ties",
              "threshold": "validation sensitivity >=0.90, maximize specificity",
              "augmentation": "none; cached deterministic frozen RAD-DINO features",
              "test_access_during_fit": "feature file integrity checks only; no test metrics or model selection"}
    write_json(out / "training_config.json", config)
    scaler = StandardScaler().fit(xtrain)
    linear = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=42)
    linear.fit(scaler.transform(xtrain), ytrain)
    if max(linear.n_iter_) >= 2000:
        raise RuntimeError("Logistic regression did not converge")
    lp = linear.predict_proba(scaler.transform(xval))[:, 1]
    write_csv(out / "linear_validation_predictions.csv", [{"image_id":r["image_id"],"label":int(y),"tb_probability":float(p)} for r,y,p in zip(valrows,yval,lp)])
    np.savez(out / "linear.npz", mean=scaler.mean_, scale=scaler.scale_,
             coef=linear.coef_, intercept=linear.intercept_)
    write_json(out / "linear.json", {"threshold": select_threshold(yval, lp),
                                    "validation_auroc": float(roc_auc_score(yval, lp)), "C": 1.0})
    for seed in seeds:
        seed_everything(seed)
        model = head_model(xtrain.shape[1])
        train = torch.utils.data.TensorDataset(torch.from_numpy(xtrain), torch.from_numpy(ytrain))
        loader = torch.utils.data.DataLoader(train, batch_size=8, shuffle=True,
                                             generator=torch.Generator().manual_seed(seed))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        weights = 1.0 / np.bincount(ytrain, minlength=2)
        weights = torch.tensor(weights / weights.sum(), dtype=torch.float32)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
        best_auc, best_epoch = -1.0, None
        log = []
        for epoch in range(1, epochs + 1):
            model.train()
            total = 0.0
            for x, y in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                total += loss.item() * len(y)
            scheduler.step()
            model.eval()
            with torch.inference_mode():
                probabilities = model(torch.from_numpy(xval)).softmax(-1)[:, 1].numpy()
            auc = float(roc_auc_score(yval, probabilities))
            log.append({"epoch": epoch, "train_loss": total / len(ytrain), "validation_auroc": auc})
            if auc > best_auc:
                best_auc, best_epoch = auc, epoch
                torch.save(model.state_dict(), out / f"head_seed{seed}.pt")
            print(f"Seed {seed}, epoch {epoch}: validation AUROC {auc:.4f}", flush=True)
        model.load_state_dict(torch.load(out / f"head_seed{seed}.pt", weights_only=True), strict=True)
        model.eval()
        with torch.inference_mode():
            p = model(torch.from_numpy(xval)).softmax(-1)[:, 1].numpy()
        write_csv(out / f"head_seed{seed}_validation_predictions.csv", [{"image_id":r["image_id"],"label":int(y),"tb_probability":float(prob)} for r,y,prob in zip(valrows,yval,p)])
        write_csv(out / f"training_seed{seed}.csv", log)
        write_json(out / f"head_seed{seed}.json", {"threshold": select_threshold(yval, p),
                   "selected_epoch": best_epoch, "validation_auroc": best_auc, "input_dim": xtrain.shape[1]})
    write_json(out / "FIT_COMPLETE.json", {"files": {p.name: digest(p) for p in sorted(out.iterdir()) if p.is_file()}})


def evaluate(study, repeats=2000):
    import torch
    rows, protocol = locked_manifest(study)
    models = Path(study) / "models"
    done = json.loads((models / "FIT_COMPLETE.json").read_text())
    for name, expected in done["files"].items():
        if digest(models / name) != expected:
            raise ValueError(f"Model artifact changed: {name}")
    config = json.loads((models / "training_config.json").read_text())
    if config["manifest_sha256"] != protocol["manifest_sha256"] or config["feature_fingerprint"] != feature_fingerprint(study, rows):
        raise ValueError("Data or cached features changed after fitting")
    results = Path(study) / "results"
    if results.exists():
        raise FileExistsError("Test evaluation already exists; preserve the original results")
    x, y, selected = load_features(study, rows, "test")
    groups = np.array([r["group_id"] for r in selected])
    results.mkdir()
    names = ["linear"] + [f"head_seed{s}" for s in config["seeds"]]
    summary = []
    all_probabilities = {}
    for name in names:
        meta = json.loads((models / f"{name}.json").read_text())
        if name == "linear":
            with np.load(models / "linear.npz", allow_pickle=False) as z:
                logits = (((x - z["mean"]) / z["scale"]) @ z["coef"].T + z["intercept"]).ravel()
            p = 1 / (1 + np.exp(-np.clip(logits, -700, 700)))
        else:
            model = head_model(meta["input_dim"])
            model.load_state_dict(torch.load(models / f"{name}.pt", weights_only=True), strict=True)
            model.eval()
            with torch.inference_mode():
                p = model(torch.from_numpy(x)).softmax(-1)[:, 1].numpy()
        all_probabilities[name] = p
        threshold = meta["threshold"]
        vals = metric_values(y, p, threshold)
        intervals = bootstrap(y, p, groups, threshold, repeats)
        record = {"model": name, "threshold": threshold, "n_images": len(y), "n_groups": len(set(groups)),
                  "metrics": vals, "cluster_bootstrap_95ci": intervals,
                  "threshold_0_5_metrics": metric_values(y, p, 0.5)}
        write_json(results / f"{name}_metrics.json", record)
        write_csv(results / f"{name}_predictions.csv", [{"image_id": r["image_id"],
                  "source": r["source"], "group_id": r["group_id"], "label": int(r["label"]),
                  "tb_probability": float(prob), "threshold": threshold,
                  "prediction": int(prob >= threshold)} for r, prob in zip(selected, p)])
        summary.append({"model": name, "n": len(y), "threshold": threshold, **vals,
                        **{f"{k}_ci_low": v[0] for k, v in intervals.items()},
                        **{f"{k}_ci_high": v[1] for k, v in intervals.items()}})
    write_csv(results / "summary.csv", summary)
    write_json(results / "evaluation_provenance.json", {"environment": metadata(),
               "manifest_sha256": protocol["manifest_sha256"], "bootstrap_repeats": repeats,
               "bootstrap_seed": 2026, "unit": "image metrics, patient/duplicate-group bootstrap",
               "warning": "Confidence intervals condition on fitted models and validation-selected thresholds. No clinical validation."})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, probabilities in all_probabilities.items():
        fpr, tpr, _ = roc_curve(y, probabilities)
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set(xlabel="False-positive rate", ylabel="Sensitivity", title="Held-out source: " + protocol["test_source"])
    ax.legend()
    fig.tight_layout()
    fig.savefig(results / "roc.png", dpi=200)
    plt.close(fig)
    print(f"Saved predictions, intervals and plots to {results}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["encode", "fit", "evaluate"])
    parser.add_argument("--study", default="study")
    parser.add_argument("--data", default="data")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    if args.command == "encode":
        encode(args.data, args.study, args.batch_size, args.allow_cpu)
    elif args.command == "fit":
        fit(args.study)
    else:
        evaluate(args.study)
