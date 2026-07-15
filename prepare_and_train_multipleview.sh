#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash prepare_and_train_multipleview.sh \
    --src /absolute/path/to/multipleview \
    --name dataset_name \
    [--repo /absolute/path/to/4DGaussians] \
    [--port 6017] \
    [--skip-train] \
    [--force]

What this script does:
  1) Convert source frames to 4DGaussians multipleview format:
       cam01/frame_00001.jpg, cam02/frame_00001.jpg, ...
  2) Run multipleviewprogress.sh to build sparse_, points3D_multipleview.ply,
     and poses_bounds_multipleview.npy
  3) Create arguments/multipleview/<name>.py from default.py if missing
  4) Launch training unless --skip-train is provided
EOF
}

SRC=""
NAME=""
REPO="/data3/isyang/Workspace/4DGaussians"
PORT="6017"
SKIP_TRAIN=0
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --src)
      SRC="${2:-}"; shift 2 ;;
    --name)
      NAME="${2:-}"; shift 2 ;;
    --repo)
      REPO="${2:-}"; shift 2 ;;
    --port)
      PORT="${2:-}"; shift 2 ;;
    --skip-train)
      SKIP_TRAIN=1; shift ;;
    --force)
      FORCE=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      usage
      exit 1 ;;
  esac
done

if [[ -z "$SRC" || -z "$NAME" ]]; then
  echo "[ERROR] --src and --name are required."
  usage
  exit 1
fi

if [[ ! -d "$SRC" ]]; then
  echo "[ERROR] Source directory not found: $SRC"
  exit 1
fi

if [[ ! -d "$REPO" ]]; then
  echo "[ERROR] Repo directory not found: $REPO"
  exit 1
fi

if [[ ! -f "$REPO/train.py" || ! -f "$REPO/multipleviewprogress.sh" ]]; then
  echo "[ERROR] Invalid repo path. Expected train.py and multipleviewprogress.sh under: $REPO"
  exit 1
fi

if ! command -v colmap >/dev/null 2>&1; then
  echo "[ERROR] 'colmap' not found in PATH."
  exit 1
fi

DEST_ROOT="$REPO/data/multipleview/$NAME"
CONFIG_PATH="$REPO/arguments/multipleview/$NAME.py"
DEFAULT_CONFIG="$REPO/arguments/multipleview/default.py"

if [[ -d "$DEST_ROOT" ]]; then
  if [[ "$FORCE" -eq 1 ]]; then
    echo "[INFO] Removing existing destination (--force): $DEST_ROOT"
    rm -rf "$DEST_ROOT"
  else
    echo "[ERROR] Destination already exists: $DEST_ROOT"
    echo "        Re-run with --force to overwrite."
    exit 1
  fi
fi

echo "[INFO] Converting source frames to 4DGaussians multipleview format..."
python - "$SRC" "$DEST_ROOT" <<'PY'
from pathlib import Path
from PIL import Image
import sys

src = Path(sys.argv[1])
dst_root = Path(sys.argv[2])
dst_root.mkdir(parents=True, exist_ok=True)

cams = sorted([d for d in src.iterdir() if d.is_dir() and d.name.startswith("cam")])
if not cams:
    raise SystemExit(f"[ERROR] No camera folders found in: {src}")

for cam_idx, cam_dir in enumerate(cams, start=1):
    out_cam = dst_root / f"cam{cam_idx:02d}"
    out_cam.mkdir(parents=True, exist_ok=True)

    frames = sorted([p for p in cam_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if not frames:
        raise SystemExit(f"[ERROR] No image files found in camera folder: {cam_dir}")

    for t, frame in enumerate(frames, start=1):
        out_img = out_cam / f"frame_{t:05d}.jpg"
        Image.open(frame).convert("RGB").save(out_img, quality=95)

print(f"[INFO] Converted {len(cams)} cameras into: {dst_root}")
PY

echo "[INFO] Running COLMAP + LLFF preprocessing (multipleviewprogress.sh)..."
(
  cd "$REPO"
  bash multipleviewprogress.sh "$NAME"
)

for required in "sparse_" "points3D_multipleview.ply" "poses_bounds_multipleview.npy"; do
  if [[ ! -e "$DEST_ROOT/$required" ]]; then
    echo "[ERROR] Missing preprocessing output: $DEST_ROOT/$required"
    exit 1
  fi
done

if [[ ! -f "$DEFAULT_CONFIG" ]]; then
  echo "[ERROR] Default config not found: $DEFAULT_CONFIG"
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  cp "$DEFAULT_CONFIG" "$CONFIG_PATH"
  echo "[INFO] Created config: $CONFIG_PATH"
else
  echo "[INFO] Config already exists, keeping as-is: $CONFIG_PATH"
fi

if [[ "$SKIP_TRAIN" -eq 1 ]]; then
  echo "[DONE] Preprocessing complete. Training skipped (--skip-train)."
  echo "       Run manually:"
  echo "       cd \"$REPO\" && python train.py -s \"data/multipleview/$NAME\" --port \"$PORT\" --expname \"multipleview/$NAME\" --configs \"arguments/multipleview/$NAME.py\""
  exit 0
fi

echo "[INFO] Starting training..."
(
  cd "$REPO"
  python train.py \
    -s "data/multipleview/$NAME" \
    --port "$PORT" \
    --expname "multipleview/$NAME" \
    --configs "arguments/multipleview/$NAME.py"
)

echo "[DONE] Training completed."
