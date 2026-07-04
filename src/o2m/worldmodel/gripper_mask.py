"""Per-frame gripper masking for the wrist camera (tracks open/close).

The static bottom-centre trapezoid (:class:`.wrist_warp.GripperMask`) cannot
follow the fingers, which sweep from the centre (closed) to the bottom corners
(open). Depth alone cannot either — Video-Depth-Anything ranks the thin dark
prongs unreliably against close scenes (measured: at close range it masked up
to 70% of the frame). What actually separates the gripper is **rigidity to the
camera**.

Prefer :class:`MaskBankGripperMasker` when the dataset has several episodes:
it exploits rigidity with NO embodiment priors (state-indexed masks from
pose-diverse cross-episode frames; ~5 ms/frame after a one-time ~3 s build)
and ports to any wrist-cam robot that logs a gripper state. The
single-episode :class:`TemporalGripperMasker` below is the fallback; it needs
the Piper-tuned gates, exploited state-conditionally:

- always : *match-fraction template* — a pixel that looks the same in >=``frac``
  of frames sampled uniformly over the WHOLE episode is camera-rigid (gripper
  base). The bag handle fails this (stable only during the ~15% hover).
- closed : same test against closed-state frames only (they span the whole
  episode -> time-diverse -> fingers at centre match, scene does not).
- open   : open frames exist only during the hover, so templates cannot reject
  the (also quasi-static) handle there. Instead: *flow rigidity* — with a
  partner frame >=4 mm of EE motion away, prongs show ~0 px optical flow while
  scene at 20 cm shows ~10-20 px. Flow is only trusted where the image has
  texture (Farneback returns ~0 on the textureless bag interior).

Gates: dark (the Piper gripper is near-black), bottom 55% of the image,
connected components touching the bottom border, morphological close + dilate.

Tuned on episode_000 (`renders/worldmodel/gripper_mask_demo*.png`, rounds 1-4).
"""
from __future__ import annotations

import abc
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from ..utils.logging import get_logger

log = get_logger(__name__)


class GripperMasker(abc.ABC):
    """Per-frame gripper mask provider (the abstraction other code depends on).

    Implementations: :class:`StaticGripperMasker` (region prior, e.g. the
    trapezoid), :class:`TemporalGripperMasker` (single-episode rigidity,
    embodiment-tuned gates), :class:`MaskBankGripperMasker` (cross-episode
    state-indexed bank — no appearance/position priors, the portable one).
    A URDF-projection masker becomes possible once the wrist hand-eye
    calibration lands (see notes 2026-07-02).
    """

    @abc.abstractmethod
    def mask(self, i: int, rgb: Optional[np.ndarray] = None) -> np.ndarray:
        """HxW bool gripper mask for episode frame ``i``."""

    def masks(self, ids: Sequence[int],
              rgbs: Optional[Sequence[np.ndarray]] = None) -> List[np.ndarray]:
        return [self.mask(i, rgbs[k] if rgbs is not None else None)
                for k, i in enumerate(ids)]


class StaticGripperMasker(GripperMasker):
    """A fixed region for every frame (e.g. the legacy bottom-centre trapezoid)."""

    def __init__(self, region: np.ndarray):
        self.region = region.astype(bool)

    def mask(self, i: int, rgb: Optional[np.ndarray] = None) -> np.ndarray:
        return self.region


class MaskBankGripperMasker(GripperMasker):
    """State-indexed gripper masks from CROSS-EPISODE same-state frames.

    A 1-DoF gripper rigidly carrying the camera is a deterministic image
    function of its opening angle, so one mask per angle bucket suffices. The
    bank is built from frames with that angle gathered ACROSS EPISODES: the
    scene behind the gripper differs episode-to-episode (teleop variance moves
    the camera by cm => tens of px at scene depth, while the gripper moves 0),
    so per-pixel agreement across the references isolates the gripper with
    **no colour, position, or flow priors** — the portable replacement for the
    single-episode :class:`TemporalGripperMasker` and its tuned gates.

    Build once per embodiment/dataset (:meth:`build`), persist with
    :meth:`save`, and reuse via :meth:`load` + :meth:`attach`.

    Per-frame mask = bank[bucket(angle_i)] AND |frame_i - template_bucket| <
    tau (the current-frame check guards against occlusions of the gripper).
    """

    def __init__(self, templates: np.ndarray, cores: np.ndarray,
                 bucket_edges: np.ndarray, tau: float = 28.0,
                 dilate: int = 9, min_area: int = 120):
        self.templates = templates          # (B,H,W,3) float32 median per bucket
        self.cores = cores                  # (B,H,W) bool rigid-agreement per bucket
        self.bucket_edges = bucket_edges    # (B+1,) normalised-opening edges
        self.tau, self.dilate, self.min_area = tau, dilate, min_area
        self._open_frac: Optional[np.ndarray] = None
        self._load = None

    # -- construction ---------------------------------------------------------
    @classmethod
    def build(cls, episode_dirs: Sequence[Path], n_buckets: int = 5,
              refs_per_bucket: int = 24, tau: float = 18.0,
              agree_frac: float = 0.78, wrist_dir: str = "realsense_color",
              arm: str = "slave", **kw) -> "MaskBankGripperMasker":
        """Build the bank from all episodes of one embodiment/setup.

        Reference frames within a bucket are chosen by FARTHEST-POINT sampling
        on the EE position: same-task episodes pass through the same poses at
        the same gripper state, so naive sampling gives references whose SCENE
        also agrees (it leaked badly on pick_bag_joe). Pose diversity enforces
        the assumption the bank rests on. ``tau``/``agree_frac`` are strict:
        the camera-rigid gripper matches pixel-exactly, similar-pose scene
        only loosely.
        """
        from PIL import Image
        from ..data import Episode, load_ee_trajectory

        # Gather (paths, opening, ee positions) over all episodes; normalise
        # opening per dataset so bucket edges mean the same thing everywhere.
        entries = []
        for d in episode_dirs:
            try:
                ep = Episode(d, wrist_dir=wrist_dir)
                traj = load_ee_trajectory(ep.actions_df(), arm=arm)
                paths = ep.wrist_frames()
            except Exception as e:
                log.warning("Mask bank: skipping %s (%s)", d, e)
                continue
            n = min(len(paths), len(traj.gripper))
            entries.append((paths[:n], np.abs(traj.gripper[:n]),
                            traj.positions[:n]))
        if not entries:
            raise ValueError("Mask bank: no usable episodes.")
        g_max = max(float(g.max()) for _, g, _ in entries)
        edges = np.linspace(0.0, 1.0 + 1e-9, n_buckets + 1)

        templates, cores = [], []
        for b in range(n_buckets):
            lo, hi = edges[b], edges[b + 1]
            pool, pool_pos = [], []          # (episode_idx, frame_idx), (3,)
            for e_idx, (paths, g, pos) in enumerate(entries):
                o = g / g_max
                for f in np.where((o >= lo) & (o < hi))[0]:
                    pool.append((e_idx, int(f)))
                    pool_pos.append(pos[f])
            if len(pool) < 6:
                templates.append(None)
                cores.append(None)
                continue
            # Farthest-point sampling on EE position -> pose-diverse refs.
            pool_pos = np.stack(pool_pos)
            sel_ids = [int(np.argmin(np.linalg.norm(
                pool_pos - pool_pos.mean(0), axis=1)))]
            dmin = np.linalg.norm(pool_pos - pool_pos[sel_ids[0]], axis=1)
            while len(sel_ids) < min(refs_per_bucket, len(pool)):
                nxt = int(np.argmax(dmin))
                sel_ids.append(nxt)
                dmin = np.minimum(dmin, np.linalg.norm(pool_pos - pool_pos[nxt],
                                                       axis=1))
            sel = [pool[j] for j in sel_ids]
            spread = float(np.median(dmin))
            stack = np.stack([
                np.asarray(Image.open(entries[e][0][f]).convert("RGB"))
                for e, f in sel]).astype(np.float32)
            T = np.median(stack, axis=0)
            agree = (np.abs(stack - T[None]).mean(-1) < tau).mean(0)
            templates.append(T)
            cores.append(agree >= agree_frac)
            log.info("Mask bank bucket %d [%.2f,%.2f): %d refs (%d episodes, "
                     "pose spread %.1fcm), core %.1f%%", b, lo, hi, len(sel),
                     len({e for e, _ in sel}), 100 * spread,
                     100 * cores[-1].mean())

        shape = next(t.shape for t in templates if t is not None)
        templates = np.stack([t if t is not None else np.zeros(shape, np.float32)
                              for t in templates])
        cores = np.stack([c if c is not None else np.zeros(shape[:2], bool)
                          for c in cores])
        return cls(templates, cores, edges, tau=tau, **kw)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, templates=self.templates, cores=self.cores,
                            bucket_edges=self.bucket_edges, tau=self.tau,
                            dilate=self.dilate, min_area=self.min_area)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "MaskBankGripperMasker":
        d = np.load(str(path))
        return cls(d["templates"], d["cores"].astype(bool), d["bucket_edges"],
                   tau=float(d["tau"]), dilate=int(d["dilate"]),
                   min_area=int(d["min_area"]))

    # -- per-episode use ------------------------------------------------------
    def attach(self, wrist_paths: Sequence[Path], gripper: np.ndarray,
               g_max: Optional[float] = None) -> "MaskBankGripperMasker":
        """Bind the target episode's frame paths + gripper stream."""
        from PIL import Image
        g = np.abs(np.asarray(gripper, float))
        self._open_frac = g / max(g_max if g_max else g.max(), 1e-9)
        paths = list(wrist_paths)
        self._load = lambda i: np.asarray(Image.open(paths[i]).convert("RGB"))
        return self

    def _bucket(self, o: float) -> int:
        return int(np.clip(np.searchsorted(self.bucket_edges, o, "right") - 1,
                           0, len(self.cores) - 1))

    def mask(self, i: int, rgb: Optional[np.ndarray] = None) -> np.ndarray:
        import cv2
        if self._open_frac is None:
            raise RuntimeError("Call attach(wrist_paths, gripper) first.")
        rgb = rgb if rgb is not None else self._load(i)
        b = self._bucket(float(self._open_frac[i]))
        core, T = self.cores[b], self.templates[b]
        f = rgb.astype(np.float32)
        if core.any():
            # Auto-exposure compensation: the wrist camera re-exposes against
            # the scene (dark bag close-up brightens the gripper), shifting
            # even rigid pixels globally. Estimate the per-channel bias on the
            # core region and remove it before matching.
            bias = np.clip(np.median((T - f)[core], axis=0), -60, 60)
            f = f + bias
        m = core & (np.abs(f - T).mean(-1) < 1.5 * self.tau)
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE,
                             np.ones((9, 9), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
        keep = np.zeros_like(m)
        for l in range(1, n):
            if stats[l, cv2.CC_STAT_AREA] >= self.min_area:
                keep[lab == l] = 1
        k = self.dilate
        return cv2.dilate(keep, np.ones((k, k), np.uint8)).astype(bool)


class TemporalGripperMasker(GripperMasker):
    """Per-frame gripper masks for one episode's wrist stream.

    Args:
        wrist_paths: all frame paths (index-aligned with the streams below).
        gripper: (N,) gripper opening (any units; normalised internally).
        ee_positions: (N,3) EE positions (m) — flow-partner selection.
        tau: per-pixel mean |RGB diff| below which two frames "match".
        frac: fraction of reference frames a pixel must match to count rigid.
        open_thr: normalised opening above which the open/flow branch is used.
        dilate: safety margin (px) around the final mask.
    """

    def __init__(self, wrist_paths: Sequence[Path], gripper: np.ndarray,
                 ee_positions: np.ndarray, n_refs: int = 30, tau: float = 28.0,
                 frac: float = 0.7, open_thr: float = 0.4, v_dark: int = 95,
                 bottom_frac: float = 0.55, flow_px: float = 1.5,
                 flow_baseline_m: float = 0.004, dilate: int = 9):
        from PIL import Image
        self.paths = list(wrist_paths)
        self.pos = np.asarray(ee_positions, float)
        self.tau, self.frac = tau, frac
        self.v_dark, self.bottom_frac = v_dark, bottom_frac
        self.flow_px, self.flow_baseline_m = flow_px, flow_baseline_m
        self.dilate = dilate

        g = np.abs(np.asarray(gripper, float))
        self.open_frac = g / max(g.max(), 1e-9)
        self.open_thr = open_thr

        n = len(self.paths)
        self._load = lambda i: np.asarray(Image.open(self.paths[i]).convert("RGB"))
        all_ids = np.linspace(0, n - 1, min(n_refs, n)).astype(int)
        closed = np.where(self.open_frac < open_thr)[0]
        if len(closed) < 5:
            closed = all_ids
        closed_ids = closed[np.linspace(0, len(closed) - 1,
                                        min(n_refs, len(closed))).astype(int)]
        self._all_refs = np.stack([self._load(j) for j in all_ids]).astype(np.float32)
        self._closed_refs = np.stack([self._load(j) for j in closed_ids]).astype(np.float32)
        h, w = self._all_refs.shape[1:3]
        self._bottom = np.zeros((h, w), bool)
        self._bottom[int(bottom_frac * h):] = True
        log.info("TemporalGripperMasker: %d refs (all) + %d refs (closed), "
                 "open frames: %d/%d", len(all_ids), len(closed_ids),
                 int((self.open_frac >= open_thr).sum()), n)

    # -- pieces ---------------------------------------------------------------
    def _frac_match(self, rgb_f: np.ndarray, refs: np.ndarray) -> np.ndarray:
        hits = (np.abs(refs - rgb_f[None]).mean(-1) < self.tau).mean(0)
        return hits >= self.frac

    def _flow_rigid(self, i: int, rgb: np.ndarray) -> np.ndarray:
        import cv2
        g0 = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        n = len(self.paths)
        j = None
        for k in list(range(i + 2, min(i + 40, n))) + \
                 list(range(i - 2, max(i - 40, -1), -1)):
            if np.linalg.norm(self.pos[k] - self.pos[i]) > self.flow_baseline_m:
                j = k
                break
        if j is None:
            return np.zeros(g0.shape, bool)
        g1 = cv2.cvtColor(self._load(j), cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 21, 3, 5, 1.2, 0)
        mag = np.linalg.norm(flow, axis=-1)
        grad = cv2.Sobel(g0, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(g0, cv2.CV_32F, 0, 1) ** 2
        textured = cv2.GaussianBlur(np.sqrt(grad), (0, 0), 4) > 8.0
        return (mag < self.flow_px) & textured

    def _clean(self, mask: np.ndarray, rgb: np.ndarray) -> np.ndarray:
        import cv2
        dark = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 2] < self.v_dark
        m = (mask & dark & self._bottom).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        _, lab = cv2.connectedComponents(m)
        keep = np.zeros_like(m)
        for l in set(np.unique(lab[-3:, :])) - {0}:
            keep[lab == l] = 1
        k = self.dilate
        return cv2.dilate(keep, np.ones((k, k), np.uint8)).astype(bool)

    # -- API ------------------------------------------------------------------
    def mask(self, i: int, rgb: Optional[np.ndarray] = None) -> np.ndarray:
        """HxW bool gripper mask for frame ``i``."""
        rgb = rgb if rgb is not None else self._load(i)
        base = self._frac_match(rgb.astype(np.float32), self._all_refs)
        if self.open_frac[i] >= self.open_thr:
            state = self._flow_rigid(i, rgb)
        else:
            state = self._frac_match(rgb.astype(np.float32), self._closed_refs)
        return self._clean(base | state, rgb)

    def masks(self, ids: Sequence[int],
              rgbs: Optional[Sequence[np.ndarray]] = None) -> List[np.ndarray]:
        return [self.mask(i, rgbs[k] if rgbs is not None else None)
                for k, i in enumerate(ids)]
