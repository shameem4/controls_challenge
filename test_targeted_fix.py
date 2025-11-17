"""Test targeted improvements on segments where pid_w_ff wins."""
from pathlib import Path
from tinyphysics import run_rollout
from tinyphysics_core.config import DATASET_PATH, DEFAULT_MODEL_PATH

# Top 5 segments where FF wins big (from previous analysis)
test_segments = ['00026', '00019', '00089', '00094', '00059']

print("Testing targeted improvements on segments where pid_w_ff wins:")
print("=" * 80)

improvements = []
for seg_name in test_segments:
    seg_path = DATASET_PATH / f"{seg_name}.csv"

    if not seg_path.exists():
        print(f"Segment {seg_name} not found, skipping...")
        continue

    # Run both controllers
    ff_result = run_rollout(seg_path, controller_type='pid_w_ff', model_path=str(DEFAULT_MODEL_PATH), debug=False)
    opt_result = run_rollout(seg_path, controller_type='pid_opt', model_path=str(DEFAULT_MODEL_PATH), debug=False)

    ff_cost = ff_result.cost['total_cost']
    opt_cost = opt_result.cost['total_cost']
    diff = opt_cost - ff_cost

    improvements.append((seg_name, ff_cost, opt_cost, diff))

    status = "OPT WINS!" if diff < 0 else "FF wins" if diff > 1.0 else "TIE"
    print(f"\nSegment {seg_name}:")
    print(f"  FF:  {ff_cost:7.2f}")
    print(f"  OPT: {opt_cost:7.2f}")
    print(f"  Diff: {diff:+7.2f}  [{status}]")

print("\n" + "=" * 80)
print("Summary:")
print("=" * 80)

total_ff = sum(x[1] for x in improvements)
total_opt = sum(x[2] for x in improvements)
total_diff = sum(x[3] for x in improvements)

print(f"Total FF cost:  {total_ff:.2f}")
print(f"Total OPT cost: {total_opt:.2f}")
print(f"Total diff:     {total_diff:+.2f}")
print(f"Average diff:   {total_diff/len(improvements):+.2f}")

wins = sum(1 for x in improvements if x[3] < 0)
losses = sum(1 for x in improvements if x[3] > 1.0)
ties = sum(1 for x in improvements if -1.0 <= x[3] <= 1.0)

print(f"\nOPT wins: {wins}/{len(improvements)}")
print(f"FF wins:  {losses}/{len(improvements)}")
print(f"Ties:     {ties}/{len(improvements)}")
