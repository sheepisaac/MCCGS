#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=${WORKSPACE:-/data3/isyang/Workspace}
VERSION=${VERSION:-final_r11}
ITERATIONS=${ITERATIONS:-14000}
PORT=${PORT:-6035}
export WORKSPACE VERSION ITERATIONS PORT

exec "${WORKSPACE}/movable_camera_cluster/4DGaussians_mcc/scripts/run_mccgs_r11_2frames.sh"
