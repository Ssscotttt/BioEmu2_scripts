#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash run_bioemu2_trajectory.sh -i INPUT.cif -o OUTPUT_DIR [options]

Options:
  -i CIF      Initial atomistic structure in mmCIF format (required)
  -o DIR      Output directory within the current working directory (required)
  -n N        Number of independent time-ordered trajectories (default: 4)
  -k STEPS    Number of sequential propagation steps per trajectory (default: 100)
  -t NS       Time lag per step, 1-1000 ns (default: 10)
  -s YAML     Matching BioEmu2 system definition, if needed (optional)
  -g GPU_ID   Required physical GPU index (only this GPU exposed to Docker)
  -c DIR      Host checkpoint directory (default: /data/ziyuan/BioEmu2)
  -e SEED     Random seed (default: 42)
  -h          Show help

Run from the project directory with the CIF(s). Input, output, and YAML paths
must be within this directory; keep bioemu2_trajectory.py next to this script.
Each output directory contains run.log and trajectory_XX/{samples.xtc,topology.cif}.
EOF
}

INPUT=""; OUTPUT=""; SYSTEM=""; N=4; K=100; T=10; GPU=""; SEED=42
CHECKPOINT_DIR="/data/ziyuan/BioEmu2"
while getopts ':i:o:n:k:t:s:g:e:c:h' opt; do
  case "$opt" in
    i) INPUT="$OPTARG" ;; o) OUTPUT="$OPTARG" ;;
    n) N="$OPTARG" ;; k) K="$OPTARG" ;; t) T="$OPTARG" ;;
    s) SYSTEM="$OPTARG" ;; g) GPU="$OPTARG" ;; e) SEED="$OPTARG" ;;
    c) CHECKPOINT_DIR="$OPTARG" ;;
    h) usage; exit 0 ;;
    :) echo "Missing argument for -$OPTARG" >&2; usage >&2; exit 2 ;;
    \?) echo "Unknown option: -$OPTARG" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -n "$INPUT" && -n "$OUTPUT" ]] || { usage >&2; exit 2; }
[[ "$N" =~ ^[1-9][0-9]*$ && "$K" =~ ^[1-9][0-9]*$ ]] || {
  echo "Error: -n and -k must be positive integers" >&2; exit 2;
}
[[ "$T" =~ ^[0-9]+([.][0-9]+)?$ ]] && awk -v t="$T" 'BEGIN {exit !(t>=1 && t<=1000)}' || {
  echo "Error: -t must be between 1 and 1000 ns" >&2; exit 2;
}
[[ "$GPU" =~ ^[0-9]+$ ]] || {
  echo "Error: -g must specify exactly one physical GPU index (e.g., -g 2)" >&2
  exit 2
}
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "Seed must be a nonnegative integer" >&2; exit 2; }

# Checkpoint files are shared across all systems and never copied into /work.
[[ -d "$CHECKPOINT_DIR" ]] || {
  echo "Error: checkpoint directory not found: $CHECKPOINT_DIR" >&2; exit 1;
}
CHECKPOINT_DIR="$(cd -- "$CHECKPOINT_DIR" && pwd -P)"
for filename in checkpoint.ckpt config.yaml; do
  [[ -r "$CHECKPOINT_DIR/$filename" ]] || {
    echo "Error: missing/unreadable $CHECKPOINT_DIR/$filename" >&2; exit 1;
  }
done

ROOT="$(pwd -P)"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
[[ -f "$SCRIPT_DIR/bioemu2_trajectory.py" ]] || {
  echo "Error: bioemu2_trajectory.py must be beside the shell launcher" >&2; exit 1;
}
resolve_project_path() {
  local resolved
  if [[ "$1" = /* ]]; then resolved="$(realpath -m -- "$1")";
  else resolved="$(realpath -m -- "$ROOT/$1")"; fi
  case "$resolved" in
    "$ROOT"/*) printf '%s' "$resolved" ;;
    *) echo "Error: path must be under $ROOT: $1" >&2; return 1 ;;
  esac
}
INPUT_ABS="$(resolve_project_path "$INPUT")"
OUTPUT_ABS="$(resolve_project_path "$OUTPUT")"
[[ -f "$INPUT_ABS" ]] || { echo "No such CIF: $INPUT_ABS" >&2; exit 1; }
if [[ -d "$OUTPUT_ABS" && -n "$(find "$OUTPUT_ABS" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Error: output directory is not empty: $OUTPUT_ABS (choose a new -o)" >&2
  exit 1
fi

args=(--input "/work/${INPUT_ABS#"$ROOT"/}" \
      --output "/work/${OUTPUT_ABS#"$ROOT"/}" \
      --trajectories "$N" --steps "$K" --lag-ns "$T" --seed "$SEED" \
      --checkpoint-path "/bioemu2-model/checkpoint.ckpt")
if [[ -n "$SYSTEM" ]]; then
  SYSTEM_ABS="$(resolve_project_path "$SYSTEM")"
  [[ -f "$SYSTEM_ABS" ]] || { echo "No such YAML: $SYSTEM_ABS" >&2; exit 1; }
  args+=(--system "/work/${SYSTEM_ABS#"$ROOT"/}")
fi

mkdir -p "$OUTPUT_ABS"
LOG="$OUTPUT_ABS/run.log"
RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
CONTAINER_NAME="bioemu2_trajectory_${RUN_ID}"

# Expose exactly ONE physical GPU to the container. It becomes cuda:0 inside.
# -n trajectories are batched together on this one GPU (not one GPU each).
docker_opts=(run --rm --gpus "device=${GPU}" --name "$CONTAINER_NAME" \
  -v "$ROOT:/work" -v "$SCRIPT_DIR:/bioemu2-launcher:ro" \
  -v "$CHECKPOINT_DIR:/bioemu2-model:ro" \
  -v bioemu2-cache:/root/.cache/huggingface -w /work)

# Absolute interpreter path bypasses the PATH reset seen in the base image.
nohup docker "${docker_opts[@]}" bioemu2:cu12 \
  /opt/conda/envs/bioemu2/bin/python -u /bioemu2-launcher/bioemu2_trajectory.py \
  "${args[@]}" > "$LOG" 2>&1 < /dev/null &

printf 'Started BioEmu2 time-ordered rollouts.\n'
printf '  Input:      %s\n  Output:     %s\n  Log:        %s\n' "$INPUT_ABS" "$OUTPUT_ABS" "$LOG"
printf '  Host GPU:   %s (only GPU exposed; cuda:0 inside)\n' "$GPU"
printf '  Checkpoint: %s\n' "$CHECKPOINT_DIR/checkpoint.ckpt"
printf '  Container:  %s\n  Host PID:   %s\n' "$CONTAINER_NAME" "$!"
printf '  Rollouts:   %s; steps/rollout: %s; lag: %s ns; nominal span: %s ns\n' \
  "$N" "$K" "$T" "$(awk -v k="$K" -v t="$T" 'BEGIN {print k*t}')"
printf 'Monitor: tail -f "%s"\n' "$LOG"
