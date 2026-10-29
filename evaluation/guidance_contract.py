"""Pure verdict logic for the Gate 0 VLM-guidance semantics probe."""

from __future__ import annotations


DEFAULT_BATCH_SIZE_ATOL = 2e-4
DEFAULT_SAMPLE_ISOLATION_ATOL = 1e-6
DEFAULT_SENSITIVITY_FLOOR = 1e-7


def evaluate_guidance_contract(
    measurements: dict,
    *,
    batch_size_atol: float = DEFAULT_BATCH_SIZE_ATOL,
    sample_isolation_atol: float = DEFAULT_SAMPLE_ISOLATION_ATOL,
    sensitivity_floor: float = DEFAULT_SENSITIVITY_FLOOR,
) -> dict:
    """Return individual checks and an aggregate PASS/FAIL verdict.

    Batch-shape changes may select different CUDA GEMM/SDPA reduction paths, so
    their numerical-equivalence tolerance is separate from the stricter
    same-batch sample-isolation tolerance. Sensitivity checks must be non-zero.
    The function is dependency-free so its decision policy can be tested locally.
    """

    batch_size_deltas = (
        float(measurements["batch_size_max_abs_delta"]),
        float(measurements["duplicate_A_batch_size_max_abs_delta"]),
        float(measurements["zero_guidance_batch_size_max_abs_delta"]),
    )
    sample_isolation_deltas = (
        float(measurements["other_sample_max_abs_delta"]),
        float(measurements["preprocessed_other_sample_max_abs_delta"]),
        float(measurements["guidance_bank_other_sample_max_abs_delta"]),
    )

    checks = {
        "G0-1_preserves_batch_axis": bool(measurements["preserves_batch_axis"]),
        "G0-2_batch_size_invariance": max(batch_size_deltas) <= batch_size_atol,
        "G0-3_sample_isolation": (
            max(sample_isolation_deltas) <= sample_isolation_atol
        ),
        "G0-4_layer_mapping_coverage": bool(measurements["layer_mapping_covers_all_blocks"]),
        "G0-5_last_all_distinction": (
            float(measurements["last_all_max_abs_delta"]) > sensitivity_floor
        ),
        "G0-6_zero_guidance_sensitivity": (
            float(measurements["zero_guidance_max_abs_delta"]) > sensitivity_floor
        ),
        "G0-6_shuffle_guidance_sensitivity": (
            float(measurements["shuffle_guidance_max_abs_delta"]) > sensitivity_floor
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "thresholds": {
            "batch_size_atol": float(batch_size_atol),
            "sample_isolation_atol": float(sample_isolation_atol),
            "sensitivity_floor": float(sensitivity_floor),
        },
    }
