from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from config import PipelineConfig
from preprocess import ImagePreprocessor
from run import TripoSRRunner
from utils import Timer, ensure_dir, save_json, safe_stem


# Default export path: inVISION repo root / model.obj (parent of TripoSR/)
_DEFAULT_MODEL_OUT = Path(__file__).resolve().parent.parent / "model.obj"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Hybrid TripoSR pipeline with preprocessing, helper-image retrieval, "
            "and conservative silhouette refinement."
        )
    )
    parser.add_argument("image", type=str, help="Path to the main input image.")
    parser.add_argument("--output-dir", type=str, default="output_hybrid")
    parser.add_argument(
        "--model-out",
        type=str,
        default=str(_DEFAULT_MODEL_OUT),
        help="Path where the reconstructed mesh is always copied as the canonical model.obj.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--mc-resolution", type=int, default=384)
    parser.add_argument("--chunk-size", type=int, default=8192)

    parser.add_argument("--foreground-ratio", type=float, default=0.92)
    parser.add_argument("--contrast", type=float, default=1.08)
    parser.add_argument("--sharpness", type=float, default=1.12)

    # parser.add_argument(
    #     "--provider",
    #     type=str,
    #     default="google_vision_web",
    #     choices=["google_vision_web", "serpapi_google_lens"],
    #     help="Helper-image retrieval provider.",
    # )
    # parser.add_argument("--disable-retrieval", action="store_true")
    # parser.add_argument("--disable-refine", action="store_true")

    # parser.add_argument("--max-results", type=int, default=12)
    # parser.add_argument("--max-downloads", type=int, default=6)
    # parser.add_argument("--max-helpers", type=int, default=3)

    # parser.add_argument("--min-exact-score", type=float, default=0.72)
    # parser.add_argument("--min-view-novelty", type=float, default=0.08)

    # parser.add_argument(
    #     "--public-image-url",
    #     type=str,
    #     default=None,
    #     help="Public URL of the preprocessed main image. Needed for SerpApi provider.",
    # )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    # if args.public_image_url:
    #     os.environ["REVERSE_IMAGE_PUBLIC_URL"] = args.public_image_url

    cfg = PipelineConfig()
    cfg.reconstruction.device = args.device
    cfg.reconstruction.mc_resolution = args.mc_resolution
    cfg.reconstruction.chunk_size = args.chunk_size

    cfg.preprocess.foreground_ratio = args.foreground_ratio
    cfg.preprocess.contrast = args.contrast
    cfg.preprocess.sharpness = args.sharpness

    # cfg.retrieval.provider = args.provider
    # cfg.retrieval.enabled = not args.disable_retrieval
    # cfg.retrieval.max_downloads = args.max_downloads
    # cfg.retrieval.max_helpers = args.max_helpers
    # cfg.retrieval.min_exact_score = args.min_exact_score
    # cfg.retrieval.min_view_novelty = args.min_view_novelty

    # cfg.refinement.enabled = not args.disable_refine

    timer = Timer()
    root = ensure_dir(Path(args.output_dir) / safe_stem(args.image))
    save_json(
        root / "effective_config.json",
        {
            "preprocess": cfg.preprocess.__dict__,
            # "retrieval": cfg.retrieval.__dict__,
            "reconstruction": cfg.reconstruction.__dict__,
            # "refinement": cfg.refinement.__dict__,
        },
    )

    # 1) Preprocess main image
    timer.start("Preprocessing main image")
    preprocessor = ImagePreprocessor(cfg.preprocess)
    main_pre = preprocessor.preprocess_image(args.image, root / "main")
    timer.end("Preprocessing main image")

    # 2) Run base TripoSR
    timer.start("Running base TripoSR reconstruction")
    runner = TripoSRRunner(cfg.reconstruction)
    base = runner.run(main_pre["tripo_input_path"], root / "base_mesh")
    timer.end("Running base TripoSR reconstruction")
    
    mesh_src = Path(base["mesh_path"])
    final_mesh_path = root / "final_mesh.obj"
    final_mesh_path.write_bytes(mesh_src.read_bytes())

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    model_out.write_bytes(mesh_src.read_bytes())

    logging.info("Done. Run output mesh: %s", final_mesh_path)
    logging.info("Canonical model (inVISION root): %s", model_out.resolve())


if __name__ == "__main__":
    main()