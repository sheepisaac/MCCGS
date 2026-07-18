#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=${WORKSPACE:-/data3/isyang/Workspace}
PYTHON_BIN=${PYTHON_BIN:-python}
MCC_4DGS_DIR=${MCC_4DGS_DIR:-${WORKSPACE}/movable_camera_cluster/4DGaussians_mcc}
MCC_SCRIPTS=${MCC_SCRIPTS:-${WORKSPACE}/movable_camera_cluster/scripts}
SOURCE_IMAGES=${SOURCE_IMAGES:-${WORKSPACE}/gaussian-splatting/data/movable_camera_cluster/unity_test_02}
UNIFIED_DIR=${UNIFIED_DIR:-${MCC_SCRIPTS}/Results/unified_camera_parameters_r3}
DATASET=${DATASET:-${MCC_SCRIPTS}/4dgs_dataset/unity_test_02_r3_2frames}
VERSION=${VERSION:-r11}
ITERATIONS=${ITERATIONS:-14000}
OUTPUT=${OUTPUT:-${MCC_SCRIPTS}/4dgs_output/unity_test_02_r3_2frames_mcc_motion_${VERSION}_iter${ITERATIONS}}
CONFIG=${CONFIG:-arguments/movable_camera_cluster_2frames.py}
PORT=${PORT:-6028}

if [ ! -d "${DATASET}/sparse/0" ]; then
  echo "[${VERSION}] Creating 2-frame dataset: ${DATASET}"
  "${PYTHON_BIN}" "${MCC_SCRIPTS}/prepare_4dgs_dataset.py" \
    --unified_dir "${UNIFIED_DIR}" \
    --image_dir "${SOURCE_IMAGES}" \
    --image_subdir input \
    --output_dir "${DATASET}" \
    --num_frames 2
else
  echo "[${VERSION}] Dataset exists: ${DATASET}"
fi

cd "${MCC_4DGS_DIR}"
"${PYTHON_BIN}" train.py \
  -s "${DATASET}" \
  --model_path "${OUTPUT}" \
  --images input \
  --configs "${CONFIG}" \
  --pose_refine \
  --mcc_motion_comp \
  --iterations "${ITERATIONS}" \
  --save_iterations "${ITERATIONS}" \
  --test_iterations 3000 7000 "${ITERATIONS}" \
  --densify_until_iter 10000 \
  --mcc_motion_loss_weight 0.01 \
  --mcc_motion_start 5000 \
  --mcc_motion_end 13000 \
  --mcc_motion_interval 500 \
  --mcc_motion_sample_points 4096 \
  --mcc_motion_max_propagated_points 256 \
  --mcc_motion_confidence_threshold 0.10 \
  --mcc_proposal_min_score 0.10 \
  --mcc_proposal_quantile 0.85 \
  --mcc_proposal_topk 192 \
  --mcc_birth_opacity_scale 0.18 \
  --mcc_birth_scale_shrink 0.82 \
  --mcc_ray_birth \
  --mcc_ray_start 8000 \
  --mcc_ray_end 13000 \
  --mcc_ray_interval 1000 \
  --mcc_ray_max_points 64 \
  --mcc_ray_views 1 \
  --mcc_ray_residual_threshold 0.08 \
  --mcc_ray_quantile 0.995 \
  --mcc_ray_edge_weight 0.25 \
  --mcc_ray_depth_min 0.05 \
  --mcc_ray_depth_max 80.0 \
  --mcc_ray_depth_jitter 0.01 \
  --mcc_ray_opacity_scale 0.12 \
  --mcc_ray_scale_shrink 0.75 \
  --mcc_verify_hypotheses \
  --mcc_verify_interval 500 \
  --mcc_verify_min_tests 3 \
  --mcc_verify_views 3 \
  --mcc_verify_l1_weight 1.0 \
  --mcc_verify_edge_weight 0.25 \
  --mcc_verify_accept_threshold 0.00001 \
  --mcc_verify_reject_threshold -0.00001 \
  --port "${PORT}"
