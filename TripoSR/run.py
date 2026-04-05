from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict

import numpy as np
import open3d as o3d
import torch
import xatlas
from PIL import Image

from tsr.bake_texture import bake_texture
from tsr.system import TSR
from tsr.utils import save_video

from config import ReconstructionConfig
from utils import ensure_dir, save_json


class TripoSRRunner:
    def __init__(self, cfg: ReconstructionConfig) -> None:
        self.cfg = cfg
        self.device = cfg.device
        if not torch.cuda.is_available():
            self.device = "cpu"
        self.model = TSR.from_pretrained(
            cfg.pretrained_model_name_or_path,
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        self.model.renderer.set_chunk_size(cfg.chunk_size)
        self.model.to(self.device)

    def run(self, image_path: str, out_dir: str | Path) -> Dict:
        out_dir = ensure_dir(out_dir)
        image = Image.open(image_path).convert("RGB")

        with torch.no_grad():
            scene_codes = self.model([image], device=self.device)

        if self.cfg.render:
            render_dir = ensure_dir(out_dir / "renders")
            render_images = self.model.render(scene_codes, n_views=30, return_type="pil")
            for ri, render_image in enumerate(render_images[0]):
                render_image.save(render_dir / f"render_{ri:03d}.png")
            save_video(render_images[0], str(out_dir / "render.mp4"), fps=30)

        meshes = self.model.extract_mesh(
            scene_codes,
            not self.cfg.bake_texture,
            resolution=self.cfg.mc_resolution,
        )
        out_mesh_path = out_dir / f"mesh.{self.cfg.model_save_format}"

        if self.cfg.bake_texture:
            out_texture_path = out_dir / "texture.png"
            bake_output = bake_texture(meshes[0], self.model, scene_codes[0], self.cfg.texture_resolution)
            xatlas.export(
                str(out_mesh_path),
                meshes[0].vertices[bake_output["vmapping"]],
                bake_output["indices"],
                bake_output["uvs"],
                meshes[0].vertex_normals[bake_output["vmapping"]],
            )
            Image.fromarray((bake_output["colors"] * 255.0).astype(np.uint8)).transpose(
                Image.FLIP_TOP_BOTTOM
            ).save(out_texture_path)
        else:
            clean_mesh = clean_and_smooth_mesh(
                meshes[0],
                keep_components=self.cfg.keep_components,
                taubin_iters=self.cfg.taubin_iters,
            )
            o3d.io.write_triangle_mesh(str(out_mesh_path), clean_mesh)

        meta = {
            "input_image": str(image_path),
            "mesh_path": str(out_mesh_path),
            "device": self.device,
            "mc_resolution": self.cfg.mc_resolution,
            "chunk_size": self.cfg.chunk_size,
        }
        save_json(out_dir / "reconstruction_meta.json", meta)
        return meta


def clean_and_smooth_mesh(tri_mesh, keep_components: int = 4, taubin_iters: int = 10):
    vertices = np.asarray(tri_mesh.vertices)
    faces = np.asarray(tri_mesh.faces)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()

    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    if len(cluster_n_triangles) > 0:
        largest_ids = np.argsort(cluster_n_triangles)[::-1][:keep_components]
        keep_mask = np.isin(triangle_clusters, largest_ids)
        triangles_to_remove = np.where(~keep_mask)[0].tolist()
        mesh.remove_triangles_by_index(triangles_to_remove)
        mesh.remove_unreferenced_vertices()

    mesh = mesh.filter_smooth_taubin(number_of_iterations=taubin_iters)
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh
