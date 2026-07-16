#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=${WORKSPACE:-/data3/isyang/Workspace}
PYTHON_BIN=${PYTHON_BIN:-python}
MCC_4DGS_DIR=${MCC_4DGS_DIR:-${WORKSPACE}/movable_camera_cluster/4DGaussians_mcc}
MCC_SCRIPTS=${MCC_SCRIPTS:-${WORKSPACE}/movable_camera_cluster/scripts}
DATASET=${DATASET:-${MCC_SCRIPTS}/4dgs_dataset/unity_test_02_r3_2frames}
VERSION=${VERSION:-r7}
ITERATION=${ITERATION:-14000}
MODEL_PATH=${MODEL_PATH:-${MCC_SCRIPTS}/4dgs_output/unity_test_02_r3_2frames_mcc_motion_${VERSION}_iter${ITERATION}}
CONFIG=${CONFIG:-arguments/movable_camera_cluster_2frames.py}
METRIC_OUTPUT=${METRIC_OUTPUT:-${MCC_SCRIPTS}/metrics_log/perframe_render_metrics_mcc_motion_${VERSION}_2frames_iter${ITERATION}_test.json}
PLY_PATH=${MODEL_PATH}/point_cloud/iteration_${ITERATION}/point_cloud.ply

if [ ! -f "${PLY_PATH}" ]; then
  echo "Missing trained point cloud: ${PLY_PATH}" >&2
  echo "Training probably stopped before iteration ${ITERATION}; rerun training and check for 'loss is nan'." >&2
  exit 1
fi

(
  cd "${MCC_4DGS_DIR}"
  "${PYTHON_BIN}" render.py \
    --model_path "${MODEL_PATH}" \
    --source_path "${DATASET}" \
    --images input \
    --configs "${CONFIG}" \
    --iteration "${ITERATION}" \
    --skip_train \
    --skip_video
)

(
  cd "${WORKSPACE}"
  "${PYTHON_BIN}" "${MCC_SCRIPTS}/evaluate_perframe_render_metrics.py" \
    --model_path "${MODEL_PATH}" \
    --dataset_dir "${DATASET}" \
    --split test \
    --iteration "${ITERATION}" \
    --max_frames 2 \
    --output "${METRIC_OUTPUT}"
)

echo "Metric output: ${METRIC_OUTPUT}"
