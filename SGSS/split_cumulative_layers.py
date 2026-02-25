import os
import re
from plyfile import PlyData, PlyElement

import os
import re
from plyfile import PlyData, PlyElement

LAYER_RE = re.compile(r"^(?P<id>\d+)-L(?P<layer>[0-3])\.ply$", re.IGNORECASE)


def load_ply(path: str) -> PlyData:
    return PlyData.read(path)


def extract_vertices(ply: PlyData, start: int, end: int):
    v = ply["vertex"].data
    return v[start:end]


def save_ply(vertices, template_ply: PlyData, out_path: str):
    vertex_el = PlyElement.describe(vertices, "vertex")
    PlyData([vertex_el], text=template_ply.text).write(out_path)


def process_one_id(vid: str, layer_paths: dict, out_dir: str):
    # layer_paths: {0: path, 1: path, 2: path, 3: path}
    plys = {k: load_ply(layer_paths[k]) for k in range(4)}
    counts = {k: int(plys[k]["vertex"].count) for k in range(4)}

    for k in range(1, 4):
        if counts[k] < counts[k - 1]:
            raise ValueError(f"[{vid}] Not cumulative: L{k}({counts[k]}) < L{k-1}({counts[k-1]})")

    v0 = extract_vertices(plys[0], 0, counts[0])
    save_ply(v0, plys[0], os.path.join(out_dir, f"{vid}-L0.ply"))

    for k in range(1, 4):
        vk = extract_vertices(plys[k], counts[k - 1], counts[k])
        save_ply(vk, plys[k], os.path.join(out_dir, f"{vid}-L{k}.ply"))


def main(in_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    groups = {}  # vid -> {layer: path}
    for name in os.listdir(in_dir):
        m = LAYER_RE.match(name)
        if not m:
            continue
        vid = m.group("id")          
        layer = int(m.group("layer"))
        groups.setdefault(vid, {})[layer] = os.path.join(in_dir, name)

    for vid, layer_paths in sorted(groups.items(), key=lambda x: int(x[0])):
        if set(layer_paths.keys()) != {0, 1, 2, 3}:
            print(f"[Skip] {vid}: missing layers, has {sorted(layer_paths.keys())}")
            continue
        process_one_id(vid, layer_paths, out_dir)

    print(f"[Done] Pure layers written to: {out_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Convert cumulative {id}-{Lk}.ply to pure delta layers {id}-Lk.ply")
    parser.add_argument("--in_dir", required=True, help="dir with cumulative files like 61-{L0}.ply")
    parser.add_argument("--out_dir", required=True, help="output dir for pure files like 61-L0.ply")
    args = parser.parse_args()
    main(args.in_dir, args.out_dir)



'''

python split_cumulative_layers.py \
  --in_dir scenes/longdress_layered/layered_streaming_cuboids \
  --out_dir scenes/longdress_layered/layered_pure




'''
