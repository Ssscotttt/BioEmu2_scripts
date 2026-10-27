#!/usr/bin/env python3
"""Generate time-ordered BioEmu2 rollouts from an external mmCIF structure."""

import argparse
import json
from pathlib import Path

from bioemu2 import (
    System,
    load_model,
    map_cif_atom_order,
    save_frames,
    simulate_trajectory,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Starting mmCIF")
    parser.add_argument("--output", required=True, type=Path, help="Output directory")
    parser.add_argument("--system", type=Path, help="Optional matching system YAML")
    parser.add_argument("--checkpoint-path", required=True, type=Path, help="Local BioEmu2 checkpoint.ckpt")
    parser.add_argument("--trajectories", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lag-ns", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.trajectories < 1 or args.steps < 1:
        parser.error("--trajectories and --steps must both be positive")
    if not (1.0 <= args.lag_ns <= 1000.0):
        parser.error("--lag-ns must be between 1 and 1000 ns")
    if not args.input.is_file():
        parser.error(f"Input CIF not found: {args.input}")
    if args.system is not None and not args.system.is_file():
        parser.error(f"System YAML not found: {args.system}")

    if not args.checkpoint_path.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint_path}")
    if not (args.checkpoint_path.parent / "config.yaml").is_file():
        parser.error(f"Checkpoint config not found: {args.checkpoint_path.parent / 'config.yaml'}")

    args.output.mkdir(parents=True, exist_ok=True)

    print("Loading system definition...", flush=True)
    system = (
        System.from_yaml(str(args.system))
        if args.system is not None
        else System.from_cif(str(args.input))
    )

    print("Mapping CIF coordinates into BioEmu2 atom order...", flush=True)
    x0 = map_cif_atom_order(system, str(args.input))
    if x0.n_frames != 1:
        raise ValueError(f"Expected one starting conformation, got {x0.n_frames}")
    print(
        f"Input: {args.input}; {x0.n_atoms} atoms; "
        f"{args.trajectories} rollouts; {args.steps} steps; "
        f"{args.lag_ns:g} ns per step",
        flush=True,
    )

    print("Loading BioEmu2 model...", flush=True)
    model = load_model(checkpoint_path=args.checkpoint_path)

    print("Starting sequential propagation...", flush=True)
    # Each replicated starting frame becomes an independent, time-ordered rollout.
    # All rollouts include time zero: (steps + 1) saved frames per trajectory.
    starts = x0[[0] * args.trajectories]
    trajectories = simulate_trajectory(
        system,
        model=model,
        x0=starts,
        lag_time_ns=args.lag_ns,
        num_trajectory_steps=args.steps,
        seed=args.seed,
    )

    for i, trajectory in enumerate(trajectories, start=1):
        target = args.output / f"trajectory_{i:02d}"
        path = save_frames(trajectory, target)
        print(f"Saved trajectory {i}/{len(trajectories)}: {path}", flush=True)

    manifest = {
        "input_cif": str(args.input),
        "system_yaml": str(args.system) if args.system is not None else None,
        "num_trajectories": args.trajectories,
        "num_trajectory_steps": args.steps,
        "frames_per_trajectory": args.steps + 1,
        "lag_time_ns": args.lag_ns,
        "nominal_span_ns": args.steps * args.lag_ns,
        "seed": args.seed,
        "checkpoint_path": str(args.checkpoint_path),
        "note": "Time-ordered BioEmu2 model rollouts; not validated MD kinetics.",
    }
    (args.output / "run_info.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("All rollouts saved successfully.", flush=True)


if __name__ == "__main__":
    main()
