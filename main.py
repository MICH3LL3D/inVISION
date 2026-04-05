#!/usr/bin/env python3
"""Run TripoSR, save mesh as model.obj at the inVISION repo root, then open the hand-tracked viewer."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
TRIPOSR_WORKFLOW = REPO_ROOT / "TripoSR" / "tripo_workflow.py"
DEFAULT_MODEL = REPO_ROOT / "model.obj"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "1) Run TripoSR (tripo_workflow.py) on an image. "
            "2) Copy the result to model.obj at the inVISION root (override with --model-out). "
            "3) Launch obj_render_ext hand/gesture 3D viewer on that file."
        )
    )
    parser.add_argument(
        "image",
        type=Path,
        nargs="?",
        default=None,
        help="Input image for TripoSR (omit with --skip-reconstruct).",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"Canonical mesh path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--skip-reconstruct",
        action="store_true",
        help="Skip TripoSR; only open the viewer if model-out already exists.",
    )
    parser.add_argument("--viewer-width", type=int, default=1000)
    parser.add_argument("--viewer-height", type=int, default=700)
    parser.add_argument(
        "--face-step",
        type=int,
        default=None,
        help="Viewer face stride (omit to auto-scale from .obj file size, ~2.8 MiB → 5).",
    )
    parser.add_argument(
        "--hand-model",
        type=Path,
        default=REPO_ROOT / "hand_landmarker.task",
        help="Path to hand_landmarker.task (MediaPipe).",
    )

    args, triposr_extra = parser.parse_known_args()

    if not args.skip_reconstruct:
        if args.image is None:
            sys.exit("Pass an input image, or use --skip-reconstruct to only open the viewer.")
        image = args.image.expanduser().resolve()
        if not image.is_file():
            sys.exit(f"Input image not found: {image}")

    model_out = args.model_out.expanduser().resolve()

    if not TRIPOSR_WORKFLOW.is_file():
        sys.exit(f"TripoSR workflow not found: {TRIPOSR_WORKFLOW}")

    if not args.skip_reconstruct:
        cmd = [
            sys.executable,
            str(TRIPOSR_WORKFLOW),
            str(image),
            "--model-out",
            str(model_out),
            *triposr_extra,
        ]
        subprocess.run(cmd, cwd=str(REPO_ROOT / "TripoSR"), check=True)

    if not model_out.is_file():
        sys.exit(f"No mesh at {model_out}. Run without --skip-reconstruct first.")
    # model_out = DEFAULT_MODEL
    sys.path.insert(0, str(REPO_ROOT))
    from obj_render_ext import ObjectViewerApp

    app = ObjectViewerApp(
        obj_path=str(model_out),
        width=args.viewer_width,
        height=args.viewer_height,
        face_step=args.face_step,
        model_path=str(args.hand_model),
    )
    app.run()


if __name__ == "__main__":
    main()
