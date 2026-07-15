# MCCGS: Movable Camera Cluster Gaussian Splatting

MCCGS is a research prototype for reconstructing dynamic scenes captured by a **movable camera cluster** into a unified 4D Gaussian representation.

This repository is based on [4DGaussians](https://github.com/hustvl/4DGaussians), with experimental changes for movable multi-camera capture, frame-aware timestamps, pose refinement, and posterior-guided Gaussian completion.

> Status: early research code. The implementation is intended for rapid experiments and ablations rather than polished general-purpose use.

## Motivation

Standard 4D Gaussian Splatting assumes that the dynamic scene can be represented from available observations with a shared canonical Gaussian set and a time-dependent deformation field. In movable camera cluster capture, however, each frame and each camera view can expose different regions of the scene. This creates a harder problem:

```text
Given sparse and uneven view-time observations,
infer a globally aligned 4D Gaussian hypothesis set
that can explain both observed regions and plausible missing regions.
```

MCCGS explores this as a posterior-guided Gaussian hypothesis problem:

- Existing Gaussians explain observed view-time evidence.
- Unexplained residuals and uncertain motion indicate missing hypotheses.
- New Gaussians are proposed conservatively as hypotheses.
- Hypotheses are optimized, retained, or pruned through multi-view/time evidence.

## Current Features

- Frame-aware timestamp parsing for COLMAP-style movable camera cluster datasets.
- Frame-level camera pose refinement.
- Motion-compensated Gaussian propagation.
- Bayesian-style posterior scoring for conservative Gaussian completion.
- Render-time loading of learned pose correction.
- Per-frame 3DGS export utilities for comparison/debugging.

The current best experimental direction is the `r5` setting:

```text
pose refinement
+ weak motion compensation
+ posterior-guided conservative Gaussian birth
```

## Repository Layout

```text
arguments/
  movable_camera_cluster.py        # MCCGS experiment config
  movable_camera_cluster_pure10.py # pure 4DGS 10-frame baseline config

scene/
  dataset_readers.py               # frame-aware COLMAP reader
  cameras.py                       # camera metadata with frame_id
  gaussian_model.py                # Gaussian model and conservative birth helper

utils/
  pose_correction.py               # frame-level pose refinement
  gaussian_birth.py                # earlier MLP-assisted birth prototype
  motion_compensation.py           # posterior-guided motion/completion controller

train.py                           # training loop with MCCGS options
render.py                          # rendering with optional pose correction
export_perframe_3DGS.py            # per-time Gaussian export utility
```

## Environment

Follow the original 4DGS setup and install the rasterization submodules:

```bash
conda create -n Gaussians4D python=3.7
conda activate Gaussians4D

pip install -r requirements.txt
pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
```

The development environment used for these experiments:

```text
PyTorch 1.13.1 + CUDA 11.7
NVIDIA RTX 4090
```

## Dataset Format

MCCGS expects a COLMAP-style dataset after preprocessing:

```text
dataset_root/
  input/
    frame_0001_view_01.png
    frame_0001_view_02.png
    ...
  sparse/0/
    cameras.bin
    images.bin
    points3D.bin
```

The reader extracts `frame_XXXX` from image names and assigns all views from the same frame to the same normalized timestamp.

In the movable camera cluster workspace, datasets are produced by the external script:

```bash
python movable_camera_cluster/scripts/prepare_4dgs_dataset.py \
  --unified_dir movable_camera_cluster/scripts/Results/unified_camera_parameters_r3 \
  --image_dir /data3/isyang/Workspace/gaussian-splatting/data/movable_camera_cluster/unity_test_02 \
  --image_subdir input \
  --output_dir movable_camera_cluster/scripts/4dgs_dataset/unity_test_02_r3_10frames \
  --num_frames 10
```

## Training

Example: MCCGS `r5`-style 10-frame experiment at 14k iterations.

```bash
cd /data3/isyang/Workspace/movable_camera_cluster/4DGaussians_mcc

python train.py \
  -s /data3/isyang/Workspace/movable_camera_cluster/scripts/4dgs_dataset/unity_test_02_r3_10frames \
  --model_path /data3/isyang/Workspace/movable_camera_cluster/scripts/4dgs_output/unity_test_02_r3_10frames_mcc_motion_r5_iter14000 \
  --images input \
  --configs arguments/movable_camera_cluster.py \
  --pose_refine \
  --mcc_motion_comp \
  --iterations 14000 \
  --save_iterations 14000 \
  --test_iterations 3000 7000 14000 \
  --densify_until_iter 10000 \
  --mcc_motion_loss_weight 0.01 \
  --mcc_motion_start 8000 \
  --mcc_motion_end 13000 \
  --mcc_motion_interval 1000 \
  --mcc_motion_sample_points 1024 \
  --mcc_motion_max_propagated_points 64 \
  --mcc_motion_confidence_threshold 0.35 \
  --port 6021
```

## Rendering

```bash
cd /data3/isyang/Workspace/movable_camera_cluster/4DGaussians_mcc

python render.py \
  --model_path /data3/isyang/Workspace/movable_camera_cluster/scripts/4dgs_output/unity_test_02_r3_10frames_mcc_motion_r5_iter14000 \
  --source_path /data3/isyang/Workspace/movable_camera_cluster/scripts/4dgs_dataset/unity_test_02_r3_10frames \
  --images input \
  --configs arguments/movable_camera_cluster.py \
  --iteration 14000 \
  --skip_train \
  --skip_video
```

## Notes On The Current Method

MCCGS currently treats newly created Gaussians as tentative hypotheses:

```text
posterior score =
  learned motion confidence
  * likelihood(residual proxy, uncertainty, low opacity, motion evidence)
```

High-posterior hypotheses are propagated along estimated Gaussian motion with low initial opacity and a slightly reduced scale. This is still a first approximation. A more principled next step is to explicitly track hypothesis support and reject or accept new Gaussians based on multi-view/time residual reduction.

## Acknowledgements

This codebase is built on top of:

- [4DGaussians](https://github.com/hustvl/4DGaussians)
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)

Please cite the original 4DGS and 3DGS papers when using this repository.

## License

This repository is a derivative research prototype based on
[4DGaussians](https://github.com/hustvl/4DGaussians), which is distributed
under the Apache License 2.0. The Apache 2.0 license text is preserved in
[LICENSE.md](LICENSE.md), and upstream attribution is recorded in
[NOTICE](NOTICE).

Some dependencies and submodules originate from 3D Gaussian Splatting and
related projects. Please check the upstream licenses before redistribution or
commercial use.
