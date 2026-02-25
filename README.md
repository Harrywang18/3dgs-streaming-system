
# 3DGS-Streaming-System

## Overview

This repository provides a complete pipeline for progressive 3D Gaussian Splatting (3DGS) streaming, including:

1. **3DGS training (layered or non-layered)** using Lapis-GS  
2. **Spatial cuboid partition and optional layer partition** using SGSS  
3. **Progressive transmission and WebXR rendering** using Spark  

Pipeline:

```
Multi-view Images
      ↓
Lapis-GS (Train 3DGS)
      ↓
SGSS (Cuboid Partition + Optional Layer Split)
      ↓
Spark (Progressive Streaming & Rendering)
```


---

# Part 1 – Lapis-GS: Dataset Preparation & 3DGS Training

Directory:

```
lapis-gs/
```

## 1.1 Environment Setup

Follow instructions in:

```
lapis-gs/README.md
```

Create environment:

```bash
conda env create -f environment.yml
conda activate lapis-gs
```

## 1.2 Dataset Preparation

Optional helper script:

```bash
lapis-gs/dataset_prepare.sh
```

## 1.3 Train 3DGS

Run:

```bash
lapis-gs/train_full_pipeline.sh
```

### Non-layered 3DGS

In `train_full_pipeline.py`:

```python
resolution_scales = [2]
```

### Layered 3DGS

Modify:

```python
resolution_scales = [16, 8, 4, 2, 1]
```

## 1.4 Export Highest Resolution 3DGS

Export the highest-resolution 3DGS model (`.ply`) for cuboid partition.

---

# Part 2 – SGSS: Cuboid & Layer Partition

Directory:

```
SGSS/
```

## 2.1 Cuboid Partition

```bash
SGSS/run_all_scripts_detail.sh
```

Outputs:

```
- Cuboid PLY files
- voxel_ilp.json
```

## 2.2 Optional: Layer Partition Inside Each Cuboid

```bash
SGSS/split_cuboids_to_layers_360.sh
```

## 2.3 Prepare Data for Streaming

Example structure:

```
room_data/
    cuboid_000.ply
    cuboid_001.ply
    voxel_ilp.json
```

---

# Part 3 – Spark: Progressive Streaming & WebXR Rendering

Directory:

```
spark/
```

## 3.1 Environment Setup

Follow:

```
spark/README.md
```

Install:

```bash
npm install
```

## 3.2 Start HTTPS Server

```bash
npm run dev
```

## 3.3 Example Modes

```
spark/examples/room_cuboids
spark/examples/room_progressive
spark/examples/webxr
```

## 3.4 Configure Data Path

In `index.html`:

```javascript
const CUBOID_PLY_DIR = "your_directory/";
const CUBOID_INDEX_URL = "your_directory/voxel_ilp.json";
```

---

# Optional – Convert PLY to SOG

```bash
spark/convert_ply_to_sog.sh
```

SOG conversion environment:

https://github.com/playcanvas/splat-transform

---

# Third-Party Code Notice

Parts of this repository are derived from or based on the following open-source projects:

- Lapis-GS: https://github.com/nus-vv-streams/lapis-gs
- SGSS: https://github.com/symmru/SGSS
- Spark: https://github.com/sparkjsdev/spark

Original licenses are preserved in their respective subdirectories where applicable.

