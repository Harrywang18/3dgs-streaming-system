#!/usr/bin/env bash

set -e

for SCENE in flowers; do
  echo "[INFO] Converting PLY to SOG for scene: $SCENE"
  
  IN_DIR="examples/room_cuboids/optimal_voxels_${SCENE}"
  OUT_DIR="examples/room_cuboids/optimal_voxels_${SCENE}_sogs"

  mkdir -p "$OUT_DIR"
  shopt -s nullglob

  for ply in "$IN_DIR"/*.ply; do
    fname=$(basename "$ply")        # room-xxx-Ly.ply
    base="${fname%.ply}"            # room-xxx-Ly
    out="$OUT_DIR/${base}.sog"

    echo "[INFO] $fname -> ${base}.sog"
    splat-transform -w "$ply" "$out"
  done

  echo "[DONE] All ply files converted to sog in $OUT_DIR"
done

