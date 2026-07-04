"""MVGenMaster-based wrist-view synthesis (multi-view generative NVS).

A third wrist renderer besides depth-warp (:mod:`.wrist_warp`) and GenWarp
(:mod:`.genwarp_warp`): **MVGenMaster** (CVPR 2025) is a multi-view diffusion
model that generates target views conditioned on a *variable number of reference
views with known cameras* plus 3D priors (reference pixels warped by depth).
Where GenWarp is single-source, MVGenMaster can fuse SEVERAL real frames
(e.g. the k-th / 2k-th frames before and after, or the third-person ZED view),
which is exactly what the disocclusion problem needs.

Setup (see ``docs/mvgenmaster.md``):
  - repo cloned at ``a2l/MVGenMaster`` (this wrapper adds it to ``sys.path``;
    the repo's vendored ``my_diffusers`` 0.29 got tiny compat patches to import
    under transformers 5.x),
  - ``check_points/pretrained_model`` (config.yaml + ema_unet.pt + ratio_set.json)
    and ``check_points/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth`` from the repo's
    HF release; the SD-2.1 base is pulled from the HF cache on first load.

Geometry: everything lives in the ROBOT BASE frame, metres, OpenCV optical
convention (x-right, y-down, z-forward). Wrist camera-to-world (c2w) comes from
FK of ``hand_cam`` remapped by ``OPTICAL_FROM_LINK`` (:func:`wrist_c2w`); the
perturbed target is the same rotation with the base-frame EE offset added to the
position — identical parallax definition to the depth-warp's ``dcam``. The model
consumes w2c = inv(c2w) and per-view intrinsics, and its 3D priors need the ref
DEPTH on the SAME scale as the camera translations: use :meth:`MVGenWrapper.
dust3r_depths` (DUSt3R locked to the known metric poses -> metric depth,
recommended) or pass VDA depth with a ``depth_scale`` fudge.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .wrist_warp import OPTICAL_FROM_LINK

# Repo location (edit here if the repo moves).
_MVGEN_REPO = Path("/home/griffing52/vail/bot2bot/bot2bot/a2l/MVGenMaster")


def _ensure_path(repo: Path = _MVGEN_REPO) -> None:
    # NOTE: MVGenMaster imports itself as top-level `src`, `my_diffusers`, `dust3r`;
    # o2m is installed as `o2m` so the `src` name is free in this process.
    p = str(repo)
    if p not in sys.path:
        sys.path.insert(0, p)


def wrist_c2w(cam_T_base_link: np.ndarray) -> np.ndarray:
    """FK pose of the ``hand_cam`` LINK (base<-link, 4x4) -> optical c2w (base frame)."""
    c2w = np.eye(4)
    c2w[:3, :3] = cam_T_base_link[:3, :3] @ OPTICAL_FROM_LINK
    c2w[:3, 3] = cam_T_base_link[:3, 3]
    return c2w


def offset_target_c2w(ref_c2w: np.ndarray, base_offset: np.ndarray) -> np.ndarray:
    """Perturbed wrist camera: same orientation, position shifted in the BASE frame.

    Matches the depth-warp's pure-translation ``dcam`` for the same offset.
    """
    tar = ref_c2w.copy()
    tar[:3, 3] = tar[:3, 3] + np.asarray(base_offset, float)
    return tar


def crop_to_aspect(rgb: np.ndarray, K: np.ndarray, aspect_hw: float,
                   depth: Optional[np.ndarray] = None):
    """Centre-crop an image (e.g. the 16:9 ZED plate) to ``aspect_hw`` = H/W and fix K.

    All views in one MVGenMaster call share a single resolution, so a third-person
    ref must first match the wrist frame's aspect; the model-side resize then keeps
    the geometry consistent through the rescaled intrinsics.
    """
    h, w = rgb.shape[:2]
    K = K.copy().astype(float)
    if h / w > aspect_hw:      # too tall -> crop rows
        nh = int(round(w * aspect_hw))
        y0 = (h - nh) // 2
        rgb = rgb[y0:y0 + nh]
        depth = depth[y0:y0 + nh] if depth is not None else None
        K[1, 2] -= y0
    else:                       # too wide -> crop cols
        nw = int(round(h / aspect_hw))
        x0 = (w - nw) // 2
        rgb = rgb[:, x0:x0 + nw]
        depth = depth[:, x0:x0 + nw] if depth is not None else None
        K[0, 2] -= x0
    return (rgb, K) if depth is None else (rgb, K, depth)


class MVGenWrapper:
    def __init__(self, repo: Optional[str] = None, model_dir: Optional[str] = None,
                 num_inference_steps: int = 50, guidance_scale: float = 2.0,
                 class_label: Optional[int] = 0, seed: int = 123,
                 max_batch_views: int = 28, half: bool = True):
        self.repo = Path(repo) if repo else _MVGEN_REPO
        self.model_dir = Path(model_dir) if model_dir else self.repo / "check_points/pretrained_model"
        self.steps = num_inference_steps
        self.guidance = guidance_scale
        self.class_label = class_label
        self.seed = seed
        self.max_batch_views = max_batch_views
        self.half = half
        self._pipe = None
        self._cfg = None
        self._ratios = None
        self._dust3r = None

    # -- model loading ---------------------------------------------------------
    # stabilityai's SD-2.x repos are gone from the Hub (404, 2026); this mirror has
    # the full diffusers layout. Only tiny configs + the 335MB VAE are fetched —
    # the UNet is built from config and ema_unet.pt supplies ALL its weights.
    _SD21_MIRROR = "SfinOe/stable-diffusion-v2-1"

    def _lazy(self):
        if self._pipe is not None:
            return
        _ensure_path(self.repo)
        import torch
        from diffusers import AutoencoderKL
        from easydict import EasyDict
        from omegaconf import OmegaConf
        from my_diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_multiview import (
            StableDiffusionMultiViewPipeline)
        from src.modules.schedulers import get_diffusion_scheduler

        cfg = EasyDict(OmegaConf.load(str(self.model_dir / "config.yaml")))
        base = cfg.pretrained_model_name_or_path
        if not Path(base).is_absolute() and not Path(base).exists():
            if (self.repo / base).exists():
                base = str(self.repo / base)   # config paths are relative to the repo
            else:
                base = self._SD21_MIRROR       # released config points at a local
                from huggingface_hub import hf_hub_download   # snapshot we don't have
                hf_hub_download(base, "scheduler/scheduler_config.json")  # warm the
                # cache: the scheduler factory loads with local_files_only=True.
        cfg.pretrained_model_name_or_path = base   # scheduler factory reads this

        vae = AutoencoderKL.from_pretrained(base, subfolder="vae")
        vae.requires_grad_(False)
        unet = self._build_unet(torch, base, cfg.model_cfg)
        scheduler = get_diffusion_scheduler(cfg, name="DDIM")
        # Construct directly (vs .from_pretrained(base)): the multiview pipeline
        # never runs the text encoder (encoder_hidden_states=None), so we skip
        # downloading it entirely.
        pipe = StableDiffusionMultiViewPipeline(
            vae=vae, unet=unet, scheduler=scheduler, safety_checker=None,
            feature_extractor=None, image_encoder=None,
            requires_safety_checker=False)
        dt = torch.float16 if (self.half and torch.cuda.is_available()) else torch.float32
        vae.to(dt)
        unet.to(dt)
        self._pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        self._cfg = cfg
        ratio_set = json.load(open(self.model_dir / "ratio_set.json"))
        self._ratios = {h / w: (h, w) for h, w in ratio_set}
        self._torch = torch

    def _build_unet(self, torch, base: str, model_cfg):
        """Multiview UNet = SD-2.1 UNet config (+ the vendored ``from_pretrained``'s
        model_cfg surgery) with ALL weights from ``ema_unet.pt``. Building from
        config skips the 3.5GB SD-2.1 UNet download the upstream runner needs just
        to overwrite; the strict ``load_state_dict`` verifies the replicated
        architecture exactly matches the checkpoint."""
        import torch.nn as nn
        from my_diffusers.models import UNet2DConditionModel

        # Guard: the surgery below replicates ONLY the released model's options.
        assert model_cfg.get("coord_encoder") == "conv_in" \
            and model_cfg.get("prior_type") == "3dpe+pixel" \
            and not model_cfg.get("coord_dropout") \
            and not model_cfg.get("use_rope", False), \
            "unexpected model_cfg — port the my_diffusers from_pretrained surgery"

        config = UNet2DConditionModel.load_config(base, subfolder="unet")
        if model_cfg.get("no_text_cross_attn", False):
            config["cross_attention_dim"] = None
        if model_cfg.get("domain_dict", None) is not None:
            config["class_embed_type"] = model_cfg["class_embed_type"]
            config["num_class_embeds"] = model_cfg["num_class_embeds"]
            config["class_embeddings_concat"] = False
        unet = UNet2DConditionModel.from_config(config)

        aic = model_cfg.get("additional_in_channels", 0)
        unet.additional_in_channels = aic
        unet.coord_encoder = model_cfg.get("coord_encoder", None)
        unet.prior_type = model_cfg.get("prior_type", "3dpe")
        from src.modules.extra_encoder import ExConvEncoder2
        unet.add_conv_in = ExConvEncoder2([aic, model_cfg.get("coord_dim", 192) + 1], 320)
        if model_cfg.get("qk_norm", False):
            from my_diffusers.models.attention_processor import Attention
            for _, m in unet.named_modules():
                if isinstance(m, Attention):
                    m.norm_q = nn.LayerNorm(m.dim_head, eps=1e-5)
                    m.norm_k = nn.LayerNorm(m.dim_head, eps=1e-5)
        unet.use_rope = False

        weights = torch.load(str(self.model_dir / "ema_unet.pt"),
                             map_location="cpu", weights_only=True)
        unet.load_state_dict(weights)          # strict: verifies the architecture
        unet.requires_grad_(False)
        unet.eval()
        return unet

    def _model_hw(self, h0: int, w0: int):
        """Closest trained resolution for this aspect ratio (from ratio_set.json)."""
        r = min(self._ratios, key=lambda k: abs(k - h0 / w0))
        return self._ratios[r]

    # -- depth -------------------------------------------------------------------
    def dust3r_depths(self, ref_rgbs: Sequence[np.ndarray], ref_c2ws: Sequence[np.ndarray],
                      ref_Ks: Sequence[np.ndarray], min_conf_thr: float = 1.5,
                      niter: int = 300, return_loss: bool = False):
        """Per-ref dense depth from DUSt3R with the aligner LOCKED to the known
        metric poses (and focals) -> depth on the same metric scale as the
        extrinsics, which is what the 3D priors require. ~seconds per call.

        With ``return_loss`` also returns the final alignment loss — the health
        check for the preset poses. Well-conditioned ref sets converge to
        ~0.001-0.01; small-baseline / static refs (where the FK mount-rotation
        error dominates) blow up to 0.5+ and the depth is garbage — callers
        should widen the refs or fall back (see the pipeline's mvgen renderer).
        """
        _ensure_path(self.repo)   # dust3r only; doesn't need the diffusion model
        import tempfile

        import cv2
        import torch
        from PIL import Image
        from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
        from dust3r.image_pairs import make_pairs
        from dust3r.inference import inference
        from dust3r.model import AsymmetricCroCo3DStereo
        from dust3r.utils.image import load_images

        if self._dust3r is None:
            self._dust3r = AsymmetricCroCo3DStereo.from_pretrained(
                str(self.repo / "check_points/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")
            ).to("cuda")
        # dust3r's loader wants files; round-trip through a tmpdir.
        with tempfile.TemporaryDirectory() as td:
            files = []
            for i, rgb in enumerate(ref_rgbs):
                f = f"{td}/ref{i:02d}.png"
                Image.fromarray(rgb).save(f)
                files.append(f)
            imgs = load_images(files, size=512, square_ok=True)
            pairs = make_pairs(imgs, scene_graph="complete", prefilter=None, symmetrize=True)
            out = inference(pairs, self._dust3r, "cuda", batch_size=1)
            scene = global_aligner(out, device="cuda",
                                   mode=GlobalAlignerMode.PointCloudOptimizer)
            # Lock cameras to the KNOWN metric poses/focals; optimise geometry only.
            poses = torch.tensor(np.stack(ref_c2ws), dtype=torch.float32)
            scene.preset_pose(poses)
            new_w = int(imgs[0]["true_shape"][0, 1])
            scene.preset_focal([float(K[0, 0]) * new_w / ref_rgbs[i].shape[1]
                                for i, K in enumerate(ref_Ks)])
            scene.min_conf_thr = min_conf_thr
            loss = scene.compute_global_alignment(init="known_poses", niter=niter,
                                                  schedule="cosine", lr=0.01)
            depths, confs = scene.get_depthmaps(), scene.get_masks()
        res = []
        for i, (d, c) in enumerate(zip(depths, confs)):
            d = d.detach().cpu().numpy().astype(np.float32)
            med = float(np.median(d[c.detach().cpu().numpy()])) if bool(c.any()) else float(np.median(d))
            d[d <= 0] = med
            h0, w0 = ref_rgbs[i].shape[:2]
            res.append(cv2.resize(d, (w0, h0), interpolation=cv2.INTER_NEAREST))
        if return_loss:
            return res, float(loss)
        return res

    # -- synthesis -----------------------------------------------------------------
    def synthesize(self, ref_rgbs: Sequence[np.ndarray], ref_depths: Sequence[np.ndarray],
                   ref_c2ws: Sequence[np.ndarray], ref_Ks: Sequence[np.ndarray],
                   tar_c2ws: Sequence[np.ndarray], tar_Ks: Optional[Sequence[np.ndarray]] = None,
                   depth_scale: float = 1.0, seed: Optional[int] = None,
                   out_size: Optional[tuple] = None) -> List[np.ndarray]:
        """Generate target views from reference views with KNOWN cameras.

        Args:
            ref_rgbs: reference frames, HxWx3 uint8, all the SAME size.
            ref_depths: per-ref HxW depth on the SAME scale as the camera
                translations (metric if poses are FK; see :meth:`dust3r_depths`).
            ref_c2ws / tar_c2ws: 4x4 camera-to-world, OpenCV optical, base frame.
            ref_Ks / tar_Ks: 3x3 intrinsics at the input resolution (tar defaults
                to ``ref_Ks[0]``).
            depth_scale: multiplies ref depth (fudge dial for non-metric depth).
            out_size: (H, W) of the returned frames; default = ref size.

        Returns:
            One HxWx3 uint8 frame per target pose.
        """
        self._lazy()
        import cv2
        torch = self._torch

        h0, w0 = ref_rgbs[0].shape[:2]
        out_h, out_w = out_size or (h0, w0)
        h, w = self._model_hw(h0, w0)
        n_ref, n_tar = len(ref_rgbs), len(tar_c2ws)
        if tar_Ks is None:
            tar_Ks = [ref_Ks[0]] * n_tar

        imgs, deps, Ks = [], [], []
        for rgb, dep, K in zip(ref_rgbs, ref_depths, ref_Ks):
            im = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
            imgs.append(torch.from_numpy(im.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1))
            deps.append(torch.from_numpy(
                cv2.resize(dep.astype(np.float32) * depth_scale, (w, h),
                           interpolation=cv2.INTER_NEAREST))[None])
            Ks.append(_scale_K(K, rgb.shape[1], rgb.shape[0], w, h))
        tKs = [_scale_K(K, w0, h0, w, h) for K in tar_Ks]

        w2cs = [np.linalg.inv(np.asarray(c, float)) for c in list(ref_c2ws) + list(tar_c2ws)]
        # Same safety rescale as the reference runner: only shrinks scenes whose
        # camera span exceeds the training normalisation (ours is centimetres).
        longest = self._cfg.get("camera_longest_side", None)
        if longest:
            centers = np.stack([np.linalg.inv(m)[:3, 3] for m in w2cs])
            span = float((centers.max(0) - centers.min(0)).max())
            if span > longest:
                s = longest / span
                for m in w2cs:
                    m[:3, 3] *= s
                deps = [d * s for d in deps]

        ref_images = torch.stack(imgs)                                     # [nr,3,h,w]
        ref_depth = torch.stack(deps)                                      # [nr,1,h,w]
        intr = torch.tensor(np.stack(Ks + tKs), dtype=torch.float32)
        extr = torch.tensor(np.stack(w2cs), dtype=torch.float32)

        outs: List[np.ndarray] = []
        gen_cap = max(1, self.max_batch_views - n_ref)
        for lo in range(0, n_tar, gen_cap):
            sel = list(range(lo, min(lo + gen_cap, n_tar)))
            nframe = n_ref + len(sel)
            image = torch.cat([ref_images,
                               torch.zeros((len(sel), 3, h, w))], 0).to("cuda")
            depth = None
            if self._cfg.model_cfg.get("enable_depth", False):
                depth = torch.cat([ref_depth,
                                   torch.zeros((len(sel), 1, h, w))], 0).to("cuda")
            idx = list(range(n_ref)) + [n_ref + s for s in sel]
            import copy
            cfg = copy.deepcopy(self._cfg)
            cfg.nframe = nframe
            g = torch.Generator().manual_seed(self.seed if seed is None else seed)
            with torch.no_grad(), torch.autocast("cuda"):
                preds = self._pipe(
                    images=image, nframe=nframe, cond_num=n_ref,
                    key_rescale=1.2 if nframe > 28 else None,
                    height=h, width=w,
                    intrinsics=intr[idx].to("cuda"), extrinsics=extr[idx].to("cuda"),
                    num_inference_steps=self.steps, guidance_scale=self.guidance,
                    output_type="np", config=cfg, tag=["custom"] * nframe,
                    class_label=self.class_label, depth=depth,
                    vae=self._pipe.vae, generator=g).images
            preds = (preds[n_ref:] * 255).astype(np.uint8)
            for j in range(preds.shape[0]):
                outs.append(cv2.resize(preds[j], (out_w, out_h),
                                       interpolation=cv2.INTER_LANCZOS4))
        return outs


def _scale_K(K: np.ndarray, w_in: int, h_in: int, w_out: int, h_out: int) -> np.ndarray:
    K = np.asarray(K, float).copy()
    K[0, :] *= w_out / w_in
    K[1, :] *= h_out / h_in
    return K
