#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=${WORKSPACE:-/data3/isyang/Workspace}
VERSION=${VERSION:-final_r11}
ITERATION=${ITERATION:-14000}
export WORKSPACE VERSION ITERATION

exec "${WORKSPACE}/movable_camera_cluster/4DGaussians_mcc/scripts/render_eval_mccgs_r11_2frames.sh"
