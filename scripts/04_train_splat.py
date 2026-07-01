#!/usr/bin/env python3
"""Stage 04 — train splatfacto and export a .ply.

Auto-orient / auto-scale are disabled so the splat stays in the COLMAP frame
(required by the robot-base alignment).
"""
from __future__ import annotations

from _common import base_parser, load, log

from o2m.splat import SplatTrainer, export_ply


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--max-iters", type=int, default=None)
    p.add_argument("--no-export", action="store_true")
    args = p.parse_args()
    cfg, paths = load(args)

    trainer = SplatTrainer(
        method=cfg.get("splat.method", "splatfacto"),
        ns_train_binary=cfg.get("splat.ns_train_binary", "ns-train"),
        max_num_iterations=int(cfg.get("splat.max_num_iterations", 30000)),
        orientation_method=cfg.get("splat.dataparser.orientation_method", "none"),
        center_method=cfg.get("splat.dataparser.center_method", "none"),
        auto_scale_poses=bool(cfg.get("splat.dataparser.auto_scale_poses", False)),
        cache_images=cfg.get("splat.datamanager.cache_images", "cpu"),
        cache_images_type=cfg.get("splat.datamanager.cache_images_type", "uint8"),
    )
    config_yml = trainer.train(paths.nerfstudio, paths.splat, max_iters=args.max_iters)
    log.info("Trained splat config: %s", config_yml)

    if not args.no_export:
        # Robust point-cloud export straight from the checkpoint (no pymeshlab /
        # ns-export). Always works and gives a representation-agnostic geometry
        # handle (the Gaussian centres + colours).
        try:
            from o2m.splat import SplatModel, export_gaussian_pointcloud
            model = SplatModel.from_config(config_yml)
            pc = export_gaussian_pointcloud(model, paths.splat / "pointcloud.ply",
                                            opacity_threshold=0.1)
            log.info("Exported point cloud -> %s", pc)
        except Exception as exc:
            log.warning("Point-cloud export failed (%s).", exc)

        # Optional: nerfstudio .ply via ns-export (needs a working pymeshlab/Qt).
        if cfg.get("splat.export.use_ns_export", False):
            try:
                ply = export_ply(config_yml, paths.splat,
                                 ns_export_binary=cfg.get("splat.ns_export_binary", "ns-export"))
                log.info("Exported splat -> %s", ply)
            except Exception as exc:
                log.warning("ns-export .ply failed (%s); checkpoint+pointcloud are fine.", exc)


if __name__ == "__main__":
    main()
