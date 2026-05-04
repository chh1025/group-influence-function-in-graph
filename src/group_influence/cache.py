import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import torch


def to_jsonable(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def stable_config_hash(config, length=12):
    payload = json.dumps(to_jsonable(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def make_run_dir(cache_root, phase, config, run_id=None):
    config_hash = stable_config_hash(config)
    if run_id is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{timestamp}_{config_hash}"
    run_dir = Path(cache_root) / phase / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, config_hash


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, sort_keys=True)
        f.write("\n")


def save_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, None) for key in fieldnames})


def save_tensor(path, tensor):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(tensor.detach().cpu(), path)
