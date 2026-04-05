from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PreprocessConfig:
    target_size: int = 512
    foreground_ratio: float = 0.92
    contrast: float = 1.08
    sharpness: float = 1.12
    edge_feather_px: int = 2
    mask_close_kernel: int = 5
    mask_open_kernel: int = 3
    pad_ratio: float = 0.08
    white_bg: bool = False


# @dataclass
# class RetrievalConfig:
#     enabled: bool = True
#     provider: str = "serpapi_google_lens"  # serpapi_google_lens | serpapi_bing_reverse | google_vision_web
#     serpapi_api_key: Optional[str] = None
#     google_api_key: Optional[str] = None
#     max_candidates: int = 15
#     max_downloads: int = 6
#     request_timeout_sec: int = 20
#     min_exact_score: float = 0.72
#     min_view_novelty: float = 0.08
#     max_helpers: int = 3
#     user_agent: str = (
#         "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
#         "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36"
#     )


@dataclass
class ReconstructionConfig:
    pretrained_model_name_or_path: str = "stabilityai/TripoSR"
    device: str = "cuda:0"
    chunk_size: int = 8192
    mc_resolution: int = 384
    model_save_format: str = "obj"
    bake_texture: bool = False
    texture_resolution: int = 2048
    render: bool = False
    keep_components: int = 4
    taubin_iters: int = 10


# @dataclass
# class RefinementConfig:
#     enabled: bool = True
#     min_helpers_for_refine: int = 1
#     front_view_tags: List[str] = field(default_factory=lambda: ["front", "back", "three_quarter_front"])
#     side_view_tags: List[str] = field(default_factory=lambda: ["left", "right", "three_quarter_side"])
#     conservative_scale_blend: float = 0.35
#     min_scale: float = 0.75
#     max_scale: float = 1.35


@dataclass
class PipelineConfig:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    # retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    # refinement: RefinementConfig = field(default_factory=RefinementConfig)
