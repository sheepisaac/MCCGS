#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=${WORKSPACE:-/data3/isyang/Workspace}
PYTHON_BIN=${PYTHON_BIN:-python}
MCC_4DGS_DIR=${MCC_4DGS_DIR:-${WORKSPACE}/movable_camera_cluster/4DGaussians_mcc}
MCC_SCRIPTS=${MCC_SCRIPTS:-${WORKSPACE}/movable_camera_cluster/scripts}
SOURCE_IMAGES=${SOURCE_IMAGES:-${WORKSPACE}/gaussian-splatting/data/movable_camera_cluster/unity_test_02}
UNIFIED_DIR=${UNIFIED_DIR:-${MCC_SCRIPTS}/Results/unified_camera_parameters_r3}
DATASET=${DATASET:-${MCC_SCRIPTS}/4dgs_dataset/unity_test_02_r3_2frames}
VERSION=${VERSION:-r7}
ITERATIONS=${ITERATIONS:-14000}
OUTPUT=${OUTPUT:-${MCC_SCRIPTS}/4dgs_output/unity_test_02_r3_2frames_mcc_motion_${VERSION}_iter${ITERATIONS}}
CONFIG=${CONFIG:-arguments/movable_camera_cluster_2frames.py}
PORT=${PORT:-6022}

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
  --mcc_motion_sample_points 2048 \
  --mcc_motion_max_propagated_points 128 \
  --mcc_motion_confidence_threshold 0.22 \
  --mcc_verify_hypotheses \
  --mcc_verify_interval 500 \
  --mcc_verify_min_tests 2 \
  --mcc_verify_accept_threshold 0.00003 \
  --mcc_verify_reject_threshold -0.00003 \
  --port "${PORT}"
