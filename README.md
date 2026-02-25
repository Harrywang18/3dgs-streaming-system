
# 3DGS-Streaming-System

## Overview

This repository presents a complete end-to-end system for progressive 3D Gaussian Splatting (3DGS) streaming.  
The system integrates representation learning, spatial partitioning, and progressive transmission into a unified pipeline designed for bandwidth-adaptive and interactive rendering scenarios (e.g., WebXR).

The pipeline consists of three main components:

1. **3DGS training (layered or non-layered)** using Lapis-GS  ([original repository](https://github.com/nus-vv-streams/lapis-gs))  

2. **Spatial cuboid partition and optional intra-cuboid layer partition** using SGSS  ([original repository](https://github.com/symmru/SGSS))  

3. **Progressive transmission and WebXR rendering** using Spark  ([original repository](https://github.com/sparkjsdev/spark))

The overall workflow is illustrated below:

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

This stage trains a 3D Gaussian Splatting representation from multi-view images.  
Both single-scale (non-layered) and multi-scale (layered) configurations are supported.


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

Using script below following Lapis-GS:

```bash
lapis-gs/dataset_prepare.sh
```

After preparation, dataset should follow Lapis-GS directory structure.

## 1.3 Train 3DGS

After the dataset has been properly prepared, train the 3D Gaussian Splatting model by running:

```bash
lapis-gs/train_full_pipeline.sh
```

The Gaussian representation will be exported as a .ply file.
This .ply file will serve as the input to the spatial partitioning stage described in Part 2.

### Layered vs Non-layered Training

The representation can be configured as either single-scale or multi-scale.

To train a non-layered model, modify `train_full_pipeline.py` as follows:

```python
resolution_scales = [2]
```
To train a layered (multi-resolution) model that supports progressive transmission, use:

```python
resolution_scales = [16, 8, 4, 2, 1]
```

---

# Part 2 – SGSS: Cuboid & Layer Partition

In this stage, we spatially partition the scene into cuboids for streaming efficiency.  
For layered 3DGS representations, optional intra-cuboid layer partition can be performed to enable finer-grained progressive transmission.

After obtaining the `.ply` file from Part 1, proceed to the `SGSS/` directory to perform spatial partitioning.


## 2.1 Cuboid Partition

The cuboid partitioning process can be executed by running:

```bash
SGSS/run_all_scripts_detail.sh
```

This script performs voxelization, ILP-based spatial optimization, and cuboid extraction.

Upon completion, the following files will be generated:


```
- Cuboid PLY files
- voxel_ilp.json
```

Each `.ply` file corresponds to a spatial cuboid extracted from the original 3DGS representation.
The `voxel_ilp.json` file stores spatial indexing information and metadata required for streaming.

## 2.2 Optional: Layer Partition Inside Each Cuboid

If a layered 3DGS model was trained in Part 1, it is possible to further split each cuboid into layer-specific units by running:

```bash
SGSS/split_cuboids_to_layers_360.sh
```

This produces cuboid × layer `.ply` files, enabling more fine-grained progressive transmission.

## 2.3 Prepare Data for Streaming

Before moving to the streaming stage, gather all cuboid (or cuboid × layer) `.ply` files together with the corresponding `voxel_ilp.json` file into a single directory.
For example:

Example structure:

```
room_data/
    cuboid_000.ply
    cuboid_001.ply
    voxel_ilp.json
```

This directory will be referenced by Spark in the next stage.

---

# Part 3 – Spark: Progressive Streaming & WebXR Rendering

In the final stage, cuboid (or cuboid × layer) units are progressively transmitted and rendered using Spark.  
This enables viewport-aware streaming and interactive WebXR visualization under bandwidth constraints.



## 3.1 Environment Setup

All streaming and rendering operations are performed in the `spark/` directory.

First, install the required Node.js dependencies by executing:

```bash
npm install
```

## 3.2 Start HTTPS Server

After installation is complete, launch the HTTPS development server using:

```bash
npm run dev
```

Spark relies on HTTPS for WebXR support, so ensure that the server is properly started.




## 3.3 Example Modes

Next, open the desired example inside `spark/examples/`.

```
spark/examples/room_cuboids
spark/examples/room_progressive
spark/examples/webxr
```

All three examples are implemented using WebXR and can be executed on compatible XR devices (e.g., VR headsets) through a WebXR-enabled browser.

The difference between them lies in the streaming granularity:

- `room_cuboids` demonstrates cuboid-based streaming, where spatial cuboids are treated as the basic transmission unit.

- `room_progressive` demonstrates layer-based progressive streaming, where different resolution layers of the 3DGS representation are transmitted progressively.

- `webxr` demonstrates combined cuboid × layer streaming, where each transmission unit corresponds to a specific spatial cuboid and a specific layer. This configuration provides the finest-grained progressive transmission and is the most complete streaming mode in this repository.

The `webxr` directory therefore represents the integrated cuboid-and-layer streaming configuration and is recommended for full progressive streaming experiments.


## 3.4 Configure Data Path

Before running the example, edit the corresponding index.html file to point to the directory prepared in Part 2.

Specifically, update:

In `index.html`:

```javascript
const CUBOID_PLY_DIR = "your_directory/";
const CUBOID_INDEX_URL = "your_directory/voxel_ilp.json";
```


These variables must reference the directory containing the cuboid `.ply` files and the `voxel_ilp.json` metadata file.



---

# Optional – Convert PLY to SOG

To reduce transmission size, `.ply` files can be converted into `.sog` format using:

```bash
spark/convert_ply_to_sog.sh
```

The required environment for SOG conversion is described in:

https://github.com/playcanvas/splat-transform

After conversion, update the file paths in index.html accordingly.

---

# Third-Party Code Notice

This project integrates and extends several open-source projects, including:

- Lapis-GS: https://github.com/nus-vv-streams/lapis-gs
- SGSS: https://github.com/symmru/SGSS
- Spark: https://github.com/sparkjsdev/spark

We acknowledge the original authors for their contributions.  
Original licenses and copyright notices are preserved where applicable.

