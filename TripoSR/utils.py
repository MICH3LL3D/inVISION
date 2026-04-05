import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict

import torch


class Timer:
    def __init__(self) -> None:
        self.items: Dict[str, float] = {}
        self.time_scale = 1000.0
        self.time_unit = "ms"

    def start(self, name: str) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.items[name] = time.time()
        logging.info("%s ...", name)

    def end(self, name: str) -> float | None:
        if name not in self.items:
            return None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        delta = time.time() - self.items.pop(name)
        scaled = delta * self.time_scale
        logging.info("%s finished in %.2f%s.", name, scaled, self.time_unit)
        return scaled


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def safe_stem(path: str | Path) -> str:
    return Path(path).stem.replace(" ", "_")
