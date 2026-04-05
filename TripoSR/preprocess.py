from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import rembg
from PIL import Image, ImageEnhance

from config import PreprocessConfig
from utils import ensure_dir, save_json


class ImagePreprocessor:
    def __init__(self, cfg: PreprocessConfig) -> None:
        self.cfg = cfg
        self.rembg_session = rembg.new_session()

    def preprocess_image(self, image_path: str, out_dir: str | Path) -> Dict:
        out_dir = ensure_dir(out_dir)
        raw = Image.open(image_path).convert("RGBA")
        rgba = self._remove_background(raw)
        rgba = self._refine_alpha_edges(rgba)
        rgba = self._crop_and_pad(rgba)
        rgba = self._light_enhance(rgba)
        rgb_on_gray = self._composite_for_triposr(rgba)

        rgba_path = out_dir / "preprocessed_rgba.png"
        gray_path = out_dir / "preprocessed_for_triposr.png"
        mask_path = out_dir / "mask.png"
        rgba.save(rgba_path)
        rgb_on_gray.save(gray_path)
        Image.fromarray((np.array(rgba)[..., 3])).save(mask_path)

        meta = self._build_metadata(raw, rgba)
        save_json(out_dir / "preprocess_meta.json", meta)
        return {
            "rgba_path": str(rgba_path),
            "tripo_input_path": str(gray_path),
            "mask_path": str(mask_path),
            "meta": meta,
        }

    def _remove_background(self, img: Image.Image) -> Image.Image:
        out = rembg.remove(img, session=self.rembg_session)
        return out.convert("RGBA")

    def _refine_alpha_edges(self, rgba: Image.Image) -> Image.Image:
        arr = np.array(rgba)
        alpha = arr[..., 3]
        _, mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)

        close_k = np.ones((self.cfg.mask_close_kernel, self.cfg.mask_close_kernel), np.uint8)
        open_k = np.ones((self.cfg.mask_open_kernel, self.cfg.mask_open_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)
        if self.cfg.edge_feather_px > 0:
            mask = cv2.GaussianBlur(mask, (0, 0), self.cfg.edge_feather_px)

        arr[..., 3] = mask
        return Image.fromarray(arr, mode="RGBA")

    def _crop_and_pad(self, rgba: Image.Image) -> Image.Image:
        arr = np.array(rgba)
        alpha = arr[..., 3]
        ys, xs = np.where(alpha > 10)
        if len(xs) == 0 or len(ys) == 0:
            return rgba.resize((self.cfg.target_size, self.cfg.target_size), Image.LANCZOS)

        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        crop = arr[y0 : y1 + 1, x0 : x1 + 1]

        h, w = crop.shape[:2]
        pad = int(max(h, w) * self.cfg.pad_ratio)
        side = max(h, w) + 2 * pad
        canvas = np.zeros((side, side, 4), dtype=np.uint8)
        oy = (side - h) // 2
        ox = (side - w) // 2
        canvas[oy : oy + h, ox : ox + w] = crop
        return Image.fromarray(canvas, mode="RGBA").resize(
            (self.cfg.target_size, self.cfg.target_size), Image.LANCZOS
        )

    def _light_enhance(self, rgba: Image.Image) -> Image.Image:
        rgb = rgba.convert("RGB")
        alpha = rgba.getchannel("A")
        rgb = ImageEnhance.Contrast(rgb).enhance(self.cfg.contrast)
        rgb = ImageEnhance.Sharpness(rgb).enhance(self.cfg.sharpness)
        out = rgb.convert("RGBA")
        out.putalpha(alpha)
        return out

    def _composite_for_triposr(self, rgba: Image.Image) -> Image.Image:
        arr = np.array(rgba).astype(np.float32) / 255.0
        bg = 1.0 if self.cfg.white_bg else 0.5
        rgb = arr[..., :3] * arr[..., 3:4] + (1.0 - arr[..., 3:4]) * bg
        rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(rgb, mode="RGB")

    def _build_metadata(self, raw: Image.Image, processed: Image.Image) -> Dict:
        raw_w, raw_h = raw.size
        alpha = np.array(processed)[..., 3]
        object_pixels = int((alpha > 10).sum())
        total_pixels = int(alpha.size)
        bbox = self._alpha_bbox(alpha)
        return {
            "raw_size": [raw_w, raw_h],
            "processed_size": list(processed.size),
            "object_fraction": object_pixels / max(total_pixels, 1),
            "bbox": bbox,
            "config": asdict(self.cfg),
        }

    @staticmethod
    def _alpha_bbox(alpha: np.ndarray) -> Tuple[int, int, int, int] | None:
        ys, xs = np.where(alpha > 10)
        if len(xs) == 0 or len(ys) == 0:
            return None
        return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
