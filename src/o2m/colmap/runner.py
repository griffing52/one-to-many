"""Wrapper around COLMAP SfM.

Two backends, auto-selected:

- ``pycolmap`` (the Python API) — no system install needed; preferred when the
  ``colmap`` CLI is not on PATH;
- the ``colmap`` CLI (subprocess) — used when the binary is available.

Both run the standard feature -> match -> map sequence with foreground masks and
a single shared camera (the wrist-camera video). The resulting model is read
back with ``model_io``.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

from ..utils import get_logger

log = get_logger("o2m.colmap")


class ColmapRunner:
    # Tuned for low-texture indoor scenes (matte surfaces, blank walls): pull
    # far more features and verify more pairs than COLMAP's defaults.
    DEFAULT_SIFT = {
        "peak_threshold": 0.002,      # default 0.0067 -> accept fainter features
        "edge_threshold": 20.0,       # default 10 -> keep more edge-like features
        "max_num_features": 20000,    # default 8192
        "domain_size_pooling": True,
        "estimate_affine_shape": True,
    }

    # Looser-than-default incremental-mapper thresholds, so marginal frames in a
    # low-texture, low-parallax scene still register. Dotted keys address
    # nested option structs (e.g. options.mapper.abs_pose_min_num_inliers).
    DEFAULT_MAPPER = {
        "min_num_matches": 8,                    # default 15
        "min_model_size": 3,                     # default 10
        "mapper.init_min_num_inliers": 30,       # default 100
        "mapper.abs_pose_min_num_inliers": 12,   # default 30
        "mapper.abs_pose_min_inlier_ratio": 0.15,  # default 0.25
    }

    def __init__(self, binary: str = "colmap", camera_model: str = "OPENCV",
                 single_camera: bool = True, matcher: str = "exhaustive",
                 camera_params: Optional[str] = None, backend: str = "auto",
                 sift_options: Optional[dict] = None,
                 mapper_options: Optional[dict] = None,
                 camera_mode: Optional[str] = None):
        self.binary = binary
        self.camera_model = camera_model
        self.single_camera = single_camera
        # single | auto | per_folder | per_image; default follows single_camera.
        self.camera_mode = camera_mode or ("single" if single_camera else "auto")
        self.matcher = matcher
        self.camera_params = camera_params  # "fx,fy,cx,cy,k1,k2,p1,p2" prior
        self.backend = backend              # auto | cli | pycolmap
        self.sift_options = {**self.DEFAULT_SIFT, **(sift_options or {})}
        self.mapper_options = {**self.DEFAULT_MAPPER, **(mapper_options or {})}

    # --- backend selection -------------------------------------------------
    def _has_cli(self) -> bool:
        return shutil.which(self.binary) is not None

    def _has_pycolmap(self) -> bool:
        try:
            import pycolmap  # noqa: F401
            return True
        except Exception:
            return False

    def resolve_backend(self) -> str:
        if self.backend in ("cli", "pycolmap"):
            return self.backend
        if self._has_cli():
            return "cli"
        if self._has_pycolmap():
            return "pycolmap"
        return "none"

    def available(self) -> bool:
        return self.resolve_backend() != "none"

    def run(self, image_dir: Path, mask_dir: Optional[Path], db_path: Path,
            sparse_dir: Path) -> Path:
        """Run the full pipeline; returns the sparse model dir (sparse/0)."""
        backend = self.resolve_backend()
        if backend == "none":
            raise SystemExit(
                "No COLMAP backend available. Install `pycolmap` "
                "(pip/uv) or the `colmap` CLI binary."
            )
        log.info("COLMAP backend: %s", backend)
        image_dir, db_path, sparse_dir = Path(image_dir), Path(db_path), Path(sparse_dir)
        sparse_dir.mkdir(parents=True, exist_ok=True)
        if backend == "pycolmap":
            return self._run_pycolmap(image_dir, mask_dir, db_path, sparse_dir)
        return self._run_cli(image_dir, mask_dir, db_path, sparse_dir)

    # --- pycolmap backend --------------------------------------------------
    def _run_pycolmap(self, image_dir: Path, mask_dir: Optional[Path],
                      db_path: Path, sparse_dir: Path) -> Path:
        import pycolmap

        if db_path.exists():
            db_path.unlink()  # fresh database each run

        reader = pycolmap.ImageReaderOptions()
        reader.camera_model = self.camera_model
        if self.camera_params:
            reader.camera_params = self.camera_params
        if mask_dir is not None:
            reader.mask_path = str(mask_dir)

        camera_mode = {
            "single": pycolmap.CameraMode.SINGLE,
            "auto": pycolmap.CameraMode.AUTO,
            "per_folder": pycolmap.CameraMode.PER_FOLDER,
            "per_image": pycolmap.CameraMode.PER_IMAGE,
        }[self.camera_mode]

        ext = pycolmap.FeatureExtractionOptions()
        for key, val in self.sift_options.items():
            setattr(ext.sift, key, val)

        log.info("pycolmap: extracting features (tuned SIFT) ...")
        pycolmap.extract_features(
            database_path=str(db_path), image_path=str(image_dir),
            camera_mode=camera_mode, reader_options=reader, extraction_options=ext,
        )

        log.info("pycolmap: matching (%s) ...", self.matcher)
        if self.matcher == "sequential":
            pycolmap.match_sequential(database_path=str(db_path))
        else:
            pycolmap.match_exhaustive(database_path=str(db_path))

        log.info("pycolmap: incremental mapping ...")
        opts = pycolmap.IncrementalPipelineOptions()
        for key, val in self.mapper_options.items():
            if "." in key:
                obj, attr = key.split(".", 1)
                setattr(getattr(opts, obj), attr, val)
            else:
                setattr(opts, key, val)
        recs = pycolmap.incremental_mapping(
            database_path=str(db_path), image_path=str(image_dir),
            output_path=str(sparse_dir), options=opts,
        )
        if not recs:
            raise RuntimeError("pycolmap produced no reconstruction (check masks/overlap).")

        best_id = max(recs, key=lambda k: recs[k].num_reg_images())
        model0 = sparse_dir / "0"
        model0.mkdir(parents=True, exist_ok=True)
        recs[best_id].write(str(model0))
        return model0

    # --- CLI backend -------------------------------------------------------
    def _cli(self, *args: str) -> None:
        cmd = [self.binary, *args]
        log.info("$ %s", " ".join(cmd))
        subprocess.run(cmd, check=True)

    def _run_cli(self, image_dir: Path, mask_dir: Optional[Path],
                 db_path: Path, sparse_dir: Path) -> Path:
        feat = ["feature_extractor", "--database_path", str(db_path),
                "--image_path", str(image_dir),
                "--ImageReader.camera_model", self.camera_model,
                "--ImageReader.single_camera", "1" if self.single_camera else "0"]
        if mask_dir is not None:
            feat += ["--ImageReader.mask_path", str(mask_dir)]
        if self.camera_params:
            feat += ["--ImageReader.camera_params", self.camera_params]
        self._cli(*feat)

        matcher_cmd = "sequential_matcher" if self.matcher == "sequential" else "exhaustive_matcher"
        self._cli(matcher_cmd, "--database_path", str(db_path))

        self._cli("mapper", "--database_path", str(db_path),
                  "--image_path", str(image_dir), "--output_path", str(sparse_dir))

        model0 = sparse_dir / "0"
        if not model0.exists():
            raise RuntimeError(f"COLMAP produced no sparse model under {sparse_dir}")
        return model0
