import os
import numpy as np
import subprocess
import argparse

# Default scene->divisor mapping (used when not providing CLI args)
DEFAULT_DIRECTORY_DIVISORS = {
    'bicycle': 243,
    'bonsai': 243,
    'drjohnson': 285,
    'flowers': 144,
    'kitchen': 267,
    'playroom': 431,
    'garden': 230,
    'stump': 329,
    'train': 237,
    'treehill': 186,
    'truck': 108
}

def generate_lp_file(directory, cd_new, c_store_0and1, matrix_a, divisor):
    lp_filename = os.path.join(directory, f"{os.path.basename(directory)}.lp")
    y = cd_new.shape[0]

    with open(lp_filename, 'w') as lp_file:
        lp_file.write("Minimize\n obj: ")
        terms = []
        for i in range(y):
            term_1 = f"{cd_new[i, 0]} x_{i}"
            term_2 = f"{c_store_0and1[i, 0] / divisor} x_{i}"
            terms.append(f"{term_1} + {term_2}")
        lp_file.write(" + ".join(terms) + "\n")

        lp_file.write("\nSubject To\n")
        n = matrix_a.shape[0]
        for j in range(n):
            constraint_terms = [f"{matrix_a[j, i]} x_{i}" for i in range(y)]
            lp_file.write(f" c{j}: " + " + ".join(constraint_terms) + " = 1\n")

        lp_file.write("\nBinary\n")
        for i in range(y):
            lp_file.write(f" x_{i}\n")
        lp_file.write("\nEnd\n")

    return lp_filename

def run_gurobi(lp_file, directory):
    solution_file = os.path.join(directory, "solution.sol")
    subprocess.run(["gurobi_cl", f"ResultFile={solution_file}", lp_file], check=True)
    return solution_file

def parse_solution(solution_file, y):
    x_values = []
    with open(solution_file, 'r') as f:
        for line in f:
            if line.startswith('x_'):
                parts = line.split()
                value = float(parts[1])
                x_values.append(value)

    if len(x_values) != y:
        raise RuntimeError(f"Parsed {len(x_values)} variables, but expected y={y}. "
                           f"Check solution file format or variable naming.")
    return np.array(x_values).reshape(-1, 1)

def process_directory(directory, divisor):
    cd_new = np.load(os.path.join(directory, 'Cd_new.npy'))
    c_store_0and1 = np.load(os.path.join(directory, 'C_store_0and1.npy'))
    matrix_a = np.load(os.path.join(directory, 'matrix_A.npy'))

    lp_file = generate_lp_file(directory, cd_new, c_store_0and1, matrix_a, divisor)
    solution_file = run_gurobi(lp_file, directory)
    x_solution = parse_solution(solution_file, cd_new.shape[0])

    out_path = os.path.join(directory, 'x_solution.npy')
    np.save(out_path, x_solution)
    print(f"Solution saved to {out_path}")

def process_all_directories(root_directory, directory_divisors):
    for directory_name, divisor in directory_divisors.items():
        dir_path = os.path.join(root_directory, directory_name)
        if os.path.isdir(dir_path):
            print(f"Processing directory: {dir_path} with divisor {divisor}")
            process_directory(dir_path, divisor)
        else:
            print(f"[Skip] Not a directory: {dir_path}")

def parse_scene_tiles_pairs(pairs):
    """
    pairs: list like ["bicycle=243", "bonsai=243"] or ["bicycle:243", ...]
    returns dict {scene: tiles_int}
    """
    out = {}
    for p in pairs:
        if "=" in p:
            scene, tiles = p.split("=", 1)
        elif ":" in p:
            scene, tiles = p.split(":", 1)
        else:
            raise ValueError(f"Invalid --scene-tiles format: {p}. Use scene=tiles or scene:tiles")
        scene = scene.strip()
        tiles = int(tiles.strip())
        if tiles <= 0:
            raise ValueError(f"tiles must be > 0 for scene {scene}, got {tiles}")
        out[scene] = tiles
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="scenes",
                        help="Root directory containing scene subdirectories.")
    parser.add_argument("--scene", type=str, default=None,
                        help="Process a single scene name under --root (e.g., bicycle).")
    parser.add_argument("--tiles", type=int, default=None,
                        help="Tile count(divisor) for --scene (e.g., 243). Required if --scene is set.")
    parser.add_argument("--scene-tiles", nargs="*", default=None,
                        help="Process multiple scenes with explicit tiles. "
                             "Example: --scene-tiles bicycle=243 bonsai=243")
    parser.add_argument("--all", action="store_true",
                        help="Process all scenes using DEFAULT_DIRECTORY_DIVISORS.")

    args = parser.parse_args()

    # Priority: --scene-tiles > --scene+--tiles > --all > default(all)
    if args.scene_tiles:
        directory_divisors = parse_scene_tiles_pairs(args.scene_tiles)
        process_all_directories(args.root, directory_divisors)
        return

    if args.scene is not None or args.tiles is not None:
        if args.scene is None or args.tiles is None:
            raise ValueError("When using --scene, you must also provide --tiles (and vice versa).")
        directory_divisors = {args.scene: int(args.tiles)}
        process_all_directories(args.root, directory_divisors)
        return

    if args.all:
        process_all_directories(args.root, DEFAULT_DIRECTORY_DIVISORS)
        return

    # Default behavior: same as --all
    process_all_directories(args.root, DEFAULT_DIRECTORY_DIVISORS)

if __name__ == "__main__":
    main()
