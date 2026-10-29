"""Evaluate one world-coordinate trajectory pair from an NPZ file."""

import argparse
import json
import math

import numpy as np

from src.metrics import compute_trajectory_metrics
from src.kinematic_projection import project_prediction_median_bone_lengths


def evaluate_file(path, kproj=False):
    # Arrays: prediction and target, each [T, 52, 3], in metres.
    with np.load(path, allow_pickle=False) as arrays:
        prediction, target = arrays['prediction'], arrays['target']
    result = {'prediction': compute_trajectory_metrics(prediction, target)}
    if kproj:
        projected = project_prediction_median_bone_lengths(prediction)
        result['KProj'] = compute_trajectory_metrics(projected, target)
    # Ratios with zero target motion are undefined; emit standard JSON null.
    return {role: {key: value if math.isfinite(value) else None
                   for key, value in metrics.items()}
            for role, metrics in result.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='NPZ with prediction and target arrays')
    parser.add_argument('--kproj', action='store_true', help='Also evaluate prediction-only bone projection')
    args = parser.parse_args()
    print(json.dumps(evaluate_file(args.input, args.kproj), indent=2, allow_nan=False))
