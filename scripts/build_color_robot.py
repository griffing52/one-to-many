#!/usr/bin/env python3
"""Build a COLOURED MuJoCo model of the Piper arm.

The stock render model (`piper_description.urdf` / `mujoco_model/piper_description.xml`)
paints every link the same flat pale-grey. This writes a coloured MJCF using the
**STL meshes** (correct, connected geometry — the same meshes the working grey arm
uses) with a per-link Piper colour scheme applied as geom ``rgba``.

Why not the OBJ+MTL meshes under ``meshes/obj/``? Those are not in the per-link
URDF frames (e.g. the ``link6`` OBJ is 0.24 m long vs the 35 mm STL disc, and its
part decomposition differs from the URDF links), so mapping them one-to-one onto
the link bodies makes the arm fly apart. Per-link solid colours on the STL avoids
that entirely. (Finer within-link colour would need correctly-framed per-link OBJs.)

Output:  mujoco_model/piper_description_color.xml
The renderer loads it because ``o2m.robot.build_mujoco_with_camera`` accepts ``.xml``;
point ``robot.render_urdf`` (configs/robot.yaml) at it.

Usage:  python scripts/build_color_robot.py [--pkg /path/to/piper_description]
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

# Per-link RGB (AgileX Piper: white segments, black base / wrist / gripper).
WHITE = (0.86, 0.86, 0.87)
BLACK = (0.13, 0.13, 0.14)
DARK = (0.20, 0.20, 0.21)
LINK_COLORS = {
    "base_link": BLACK, "link1": WHITE, "link2": WHITE, "link3": WHITE,
    "link4": WHITE, "link5": WHITE, "link6": BLACK, "gripper_base": BLACK,
    "link7": DARK, "link8": DARK,
}


def build_mjcf(pkg: Path) -> Path:
    template = pkg / "mujoco_model/piper_description.xml"
    dst = pkg / "mujoco_model/piper_description_color.xml"
    root = ET.parse(template).getroot()

    # brighten so the white segments read as white, not grey
    vis = root.find("visual") or ET.SubElement(root, "visual")
    hl = vis.find("headlight") or ET.SubElement(vis, "headlight")
    hl.set("ambient", "0.45 0.45 0.45")
    hl.set("diffuse", "0.6 0.6 0.6")
    hl.set("specular", "0.15 0.15 0.15")

    # colour each geom by the mesh (link) it draws
    for geom in root.iter("geom"):
        color = LINK_COLORS.get(geom.get("mesh"))
        if color is not None:
            geom.set("rgba", f"{color[0]} {color[1]} {color[2]} 1")
    dst.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkg", default="/home/griffing52/vail/bot2bot/urdf/piper_description",
                    help="piper_description package dir (has mujoco_model/ + meshes/).")
    args = ap.parse_args()
    dst = build_mjcf(Path(args.pkg))
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(dst))
    print(f"wrote {dst}\ncompiled OK: ngeom={m.ngeom} nmesh={m.nmesh}")


if __name__ == "__main__":
    main()
