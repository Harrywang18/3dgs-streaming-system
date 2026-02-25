#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${1:-model/nerf_synthetic}"
SCENES_CSV="${2:-lego,chair,drums,ficus,hotdog,materials,mic,ship}"
RESS_CSV="${3:-1,2,4,8}"

OUT_CSV="stats.csv"
echo "scene,res,num_points,file_size_bytes,file_size_human" > "$OUT_CSV"

IFS=',' read -r -a SCENES <<< "$SCENES_CSV"
IFS=',' read -r -a RESS <<< "$RESS_CSV"

human_size() {
  local b=$1
  if [ "$b" -lt 1024 ]; then echo "${b}B"; return; fi
  local kb=$((b/1024))
  if [ "$kb" -lt 1024 ]; then echo "${kb}K"; return; fi
  local mb=$((kb/1024))
  if [ "$mb" -lt 1024 ]; then echo "${mb}M"; return; fi
  local gb=$((mb/1024))
  echo "${gb}G"
}

for scene in "${SCENES[@]}"; do
  for res in "${RESS[@]}"; do

    PLY="${BASE_DIR}/${scene}/lapis/${scene}_res${res}/point_cloud/iteration_30000/point_cloud.ply"
    if [ ! -f "$PLY" ]; then
      echo "WARNING: missing $PLY" >&2
      continue
    fi

    size_bytes=$(stat -c%s "$PLY")
    size_human=$(human_size "$size_bytes")

    # 尝试解析 element vertex/point N
    element_line=$(sed -n '1,/end_header/p' "$PLY" | \
  grep -a -i -m1 -E "^[[:space:]]*element[[:space:]]+(vertex|point)[[:space:]]+[0-9]+" || true)
    if [ -n "$element_line" ]; then
      num=$(echo "$element_line" | awk '{print $3}')
    else
      # ASCII 回退
      fmt_line=$(grep -m1 "^format " "$PLY" || true)
      if echo "$fmt_line" | grep -q ascii; then
        header_end=$(awk '/^end_header/ {print NR; exit}' "$PLY")
        total_lines=$(wc -l < "$PLY")
        num=$(( total_lines - header_end ))
      else
        num="UNKNOWN"
      fi
    fi

    echo "${scene},${res},${num},${size_bytes},${size_human}" >> "$OUT_CSV"
    echo "OK: ${scene} res${res} -> ${num} pts, ${size_human}"

  done
done

echo "Done. Results saved to $OUT_CSV"
