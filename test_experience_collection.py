"""Quick test to verify experience collection is working."""
from pathlib import Path
from train_adaptive_gains import AdaptiveGainTrainer, TrainingConfig
from tinyphysics_core.config import DATASET_PATH

# Get one segment
data_path = Path(DATASET_PATH)
data_paths = sorted(list(data_path.glob("*.csv")))[:1]

if len(data_paths) == 0:
    print("No data found!")
    exit(1)

print(f"Testing with segment: {data_paths[0]}")

# Create minimal config
config = TrainingConfig(
    num_epochs=1,
    episodes_per_epoch=1,
    batch_size=32,
    learning_rate=0.001,
    network_path="models/test.pkl",
    segment_limit=1,
)

# Create trainer
trainer = AdaptiveGainTrainer(config, data_paths, resume=False)

# Collect experience from one segment
print("\nCollecting experience...")
experiences, total_cost = trainer.collect_experience(data_paths[0], use_exploration=True)

print(f"\nResults:")
print(f"  Total cost: {total_cost:.4f}")
print(f"  Num experiences: {len(experiences)}")
print(f"  Buffer size before push: {len(trainer.buffer)}")

# Add to buffer
for exp in experiences:
    trainer.buffer.push(exp)

print(f"  Buffer size after push: {len(trainer.buffer)}")

if len(experiences) > 0:
    print(f"\nFirst experience:")
    features, gains, cost = experiences[0]
    print(f"  Features shape: {features.shape}")
    print(f"  Gains: P={gains[0]:.4f}, I={gains[1]:.4f}, D={gains[2]:.4f}, FF={gains[3]:.4f}")
    print(f"  Cost: {cost:.4f}")
    print("\n✅ Experience collection is WORKING!")
else:
    print("\n❌ No experiences collected - still broken!")
