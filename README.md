# MCCGS: Movable Camera Cluster Gaussian Splatting

MCCGS is a research prototype for reconstructing dynamic scenes captured by a **movable camera cluster** into a unified 4D Gaussian representation.

This repository is based on [4DGaussians](https://github.com/hustvl/4DGaussians), with experimental changes for movable multi-camera capture, frame-aware timestamps, pose refinement, posterior-guided Gaussian completion, and CVTE-based Gaussian hypothesis verification.

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
- CVTE-style held-out verification for tentative Gaussian hypotheses.
- Adaptive and edge-aware CVTE variants for detail-sensitive verification.
- Render-time loading of learned pose correction.
- Per-frame 3DGS export utilities for comparison/debugging.

The current best two-frame smoke-test direction is the `r9` setting:

```text
pose refinement
+ motion-compensated proposal
+ edge-aware CVTE verification
+ posterior-guided conservative Gaussian completion
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
  motion_compensation.py           # posterior-guided motion/CVTE controller

scripts/
  run_mccgs_r7_2frames.sh          # baseline CVTE smoke test
  run_mccgs_r8_2frames.sh          # adaptive multi-view CVTE test
  run_mccgs_r9_2frames.sh          # edge-aware CVTE test
  render_eval_mccgs_r*_2frames.sh  # render/evaluate two-frame experiments

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

Example: MCCGS `r9` two-frame edge-aware CVTE experiment at 14k iterations.

```bash
cd /data3/isyang/Workspace
conda activate Gaussians4D

bash movable_camera_cluster/4DGaussians_mcc/scripts/run_mccgs_r9_2frames.sh
```

## Rendering

```bash
cd /data3/isyang/Workspace
conda activate Gaussians4D

bash movable_camera_cluster/4DGaussians_mcc/scripts/render_eval_mccgs_r9_2frames.sh
```

## Notes On The Current Method

MCCGS treats newly created Gaussians as tentative hypotheses:

```text
posterior score =
  learned motion confidence
  * likelihood(residual proxy, uncertainty, low opacity, motion evidence)
```

High-posterior hypotheses are propagated along estimated Gaussian motion with low initial opacity and a slightly reduced scale. CVTE then evaluates these probationary Gaussians by rendering held-out views twice:

```text
R_full     = render(with probation Gaussians)
R_without  = render(with probation Gaussians hidden)
delta      = loss(R_without, GT) - loss(R_full, GT)
```

Positive delta means the probation hypotheses help the held-out render. Negative delta means they hurt and should be pruned or downweighted.

The current implementation includes three experimental variants:

- `r7`: basic CVTE smoke test with conservative hypothesis verification.
- `r8`: adaptive multi-view CVTE, using recent delta statistics to estimate a local noise floor instead of relying only on a fixed threshold.
- `r9`: edge-aware CVTE, adding image-gradient error to the verification loss so high-frequency structures such as building textures, dinosaur surfaces, and tree foliage contribute more strongly to the hypothesis score.

In the first two-frame test split, `r9` improved perceptual quality over `r7` while preserving PSNR:

```text
r7: PSNR 25.6325, SSIM 0.8993, LPIPS-VGG 0.1345, LPIPS-Alex 0.1276
r8: PSNR 24.2709, SSIM 0.8983, LPIPS-VGG 0.1351, LPIPS-Alex 0.1298
r9: PSNR 25.6416, SSIM 0.8991, LPIPS-VGG 0.1325, LPIPS-Alex 0.1253
```

These numbers are early ablation results, not final benchmark claims. They are useful mainly for comparing MCCGS hypothesis-generation variants under the same two-frame smoke-test setting.

## CVTE Options

The main CVTE-related training flags are:

```text
--mcc_verify_hypotheses
--mcc_verify_interval
--mcc_verify_min_tests
--mcc_verify_views
--mcc_verify_accept_threshold
--mcc_verify_reject_threshold
--mcc_verify_adaptive
--mcc_verify_adaptive_min_abs
--mcc_verify_adaptive_mad_scale
--mcc_verify_sign_ratio
--mcc_verify_history
--mcc_verify_l1_weight
--mcc_verify_edge_weight
```

The `r9` script enables edge-aware CVTE with:

```text
--mcc_verify_views 3
--mcc_verify_adaptive
--mcc_verify_l1_weight 1.0
--mcc_verify_edge_weight 0.25
```

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
