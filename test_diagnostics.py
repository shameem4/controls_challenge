"""Check if high-demand adaptation is triggering on FF-winning segments."""
from pathlib import Path
from tinyphysics import run_rollout
from tinyphysics_core.config import DATASET_PATH, DEFAULT_MODEL_PATH

# Top 5 segments where FF wins big
test_segments = ['00026', '00019', '00089', '00094', '00059']

print("Checking segment characteristics and adaptation triggers:")
print("=" * 80)

for seg_name in test_segments:
    seg_path = DATASET_PATH / f"{seg_name}.csv"

    if not seg_path.exists():
        print(f"Segment {seg_name} not found, skipping...")
        continue

    # Run pid_opt to get diagnostics
    opt_result = run_rollout(seg_path, controller_type='pid_opt', model_path=str(DEFAULT_MODEL_PATH), debug=False)

    diag = opt_result.diagnostics

    print(f"\nSegment {seg_name}:")
    print(f"  avg_abs_target: {diag.get('estimated_avg_abs_target_lataccel', 0):.4f} (threshold: 0.25)")
    print(f"  max_abs_target: {diag.get('max_abs_target_lataccel', 0):.4f} (threshold: 1.2)")
    print(f"  avg_target_change: {diag.get('estimated_avg_abs_target_lataccel_change', 0):.4f}")
    print(f"  velocity_target_product: {diag.get('velocity_target_product', 0):.4f}")
    print(f"  Final FF gain: {diag.get('feedforward_gain_adaptive', 0):.4f} (base: 0.75)")
    print(f"  Final P gain: {diag.get('p_adaptive', 0):.4f} (base: 0.24)")

    # Check if high-demand should trigger
    avg_target = diag.get('estimated_avg_abs_target_lataccel', 0)
    max_target = diag.get('max_abs_target_lataccel', 0)

    # Note: The controller estimates these online, so they may not exactly match the CSV
    if avg_target > 0.25 and max_target > 1.2:
        print(f"  HIGH-DEMAND MODE: SHOULD TRIGGER [YES]")
    else:
        print(f"  HIGH-DEMAND MODE: NOT TRIGGERED")
        if avg_target <= 0.25:
            print(f"    - avg_target too low ({avg_target:.4f} <= 0.25)")
        if max_target <= 1.2:
            print(f"    - max_target too low ({max_target:.4f} <= 1.2)")

print("\n" + "=" * 80)
