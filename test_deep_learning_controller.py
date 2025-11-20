"""Test script for deep learning controller."""
import sys
from pathlib import Path

# Test 1: Import and instantiate the controller
print("Test 1: Importing deep learning controller...")
try:
  from controllers.deep_learning_control import Controller as DeepLearningController
  print("[PASS] Successfully imported DeepLearningController")
except Exception as e:
  print(f"[FAIL] Failed to import: {e}")
  sys.exit(1)

# Test 2: Create controller instance without trained model
print("\nTest 2: Creating controller instance...")
try:
  controller = DeepLearningController(
    model_path="models/deep_learning_control_test.pkl",
    enable_online_learning=False
  )
  print("[PASS] Successfully created controller instance")
except Exception as e:
  print(f"[FAIL] Failed to create controller: {e}")
  import traceback
  traceback.print_exc()
  sys.exit(1)

# Test 3: Test controller reset
print("\nTest 3: Testing controller reset...")
try:
  controller.reset()
  assert len(controller.error_history) == 0
  assert len(controller.action_history) == 0
  assert controller.control_steps == 0
  print("[PASS] Controller reset successful")
except Exception as e:
  print(f"[FAIL] Reset failed: {e}")
  sys.exit(1)

# Test 4: Test compute_action with mock data
print("\nTest 4: Testing compute_action...")
try:
  from tinyphysics_core import State, FuturePlan
  from controllers import ControlState, ControlPhase

  # Create mock state
  state = State(
    roll_lataccel=0.1,
    v_ego=15.0,
    a_ego=0.5
  )

  # Create mock future plan
  future_plan = FuturePlan(
    lataccel=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0],
    roll_lataccel=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    v_ego=[15.0, 15.1, 15.2, 15.3, 15.4, 15.5, 15.6, 15.7, 15.8, 15.9],
    a_ego=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
  )

  # Create control state
  control_state = ControlState(
    phase=ControlPhase.ACTIVE,
    step_idx=0
  )

  # Compute action
  target_lataccel = 0.5
  current_lataccel = 0.0

  for i in range(15):  # Test multiple steps including warmup
    action = controller.compute_action(
      target_lataccel,
      current_lataccel,
      state,
      future_plan,
      control_state
    )

    # Simulate feedback
    controller.on_simulation_update(
      predicted_lataccel=current_lataccel + action * 0.1,
      control_state=control_state,
      step_metrics=None
    )

    current_lataccel += action * 0.1
    control_state.step_idx += 1

  assert isinstance(action, float)
  assert -3.0 <= action <= 3.0, f"Action {action} out of expected range"
  print(f"[PASS] compute_action successful, final action: {action:.3f}")

except Exception as e:
  print(f"[FAIL] compute_action failed: {e}")
  import traceback
  traceback.print_exc()
  sys.exit(1)

# Test 5: Test diagnostics
print("\nTest 5: Testing diagnostics...")
try:
  diagnostics = controller.get_diagnostics()
  assert isinstance(diagnostics, dict)
  assert 'control_steps' in diagnostics
  print(f"[PASS] Diagnostics retrieved: {diagnostics}")
except Exception as e:
  print(f"[FAIL] Diagnostics failed: {e}")
  sys.exit(1)

# Test 6: Test network creation and basic functionality
print("\nTest 6: Testing network creation...")
try:
  from deep_learning_network import DeepLearningControlNetwork
  import numpy as np

  network = DeepLearningControlNetwork(
    history_length=10,
    future_length=5,
    num_channels=7,
    conv_filters=(8, 16),
    kernel_size=3,
    fc_dims=(32, 16),
    learning_rate=0.001
  )

  # Test forward pass
  history = np.random.randn(10, 7).astype(np.float32)
  future = np.random.randn(5, 4).astype(np.float32)

  action, pred_lataccel, activations = network.forward(history, future)

  assert isinstance(action, (float, np.floating))
  assert isinstance(pred_lataccel, (float, np.floating))
  assert len(activations) > 0

  print(f"[PASS] Network forward pass successful")
  print(f"  Action: {action:.3f}, Predicted lataccel: {pred_lataccel:.3f}")

except Exception as e:
  print(f"[FAIL] Network test failed: {e}")
  import traceback
  traceback.print_exc()
  sys.exit(1)

# Test 7: Test network save/load
print("\nTest 7: Testing network save/load...")
try:
  test_model_path = "models/test_network.pkl"
  network.save(test_model_path)
  print(f"[PASS] Network saved to {test_model_path}")

  loaded_network = DeepLearningControlNetwork.load(test_model_path)
  print(f"[PASS] Network loaded from {test_model_path}")

  # Verify loaded network produces same output
  action2, pred_lataccel2, _ = loaded_network.forward(history, future)

  assert abs(action - action2) < 1e-5, f"Action mismatch: {action} vs {action2}"
  assert abs(pred_lataccel - pred_lataccel2) < 1e-5, f"Prediction mismatch: {pred_lataccel} vs {pred_lataccel2}"

  print("[PASS] Loaded network produces identical results")

  # Cleanup
  Path(test_model_path).unlink(missing_ok=True)

except Exception as e:
  print(f"[FAIL] Save/load test failed: {e}")
  import traceback
  traceback.print_exc()
  sys.exit(1)

print("\n" + "="*60)
print("All tests passed!")
print("="*60)
print("\nDeep learning controller is ready to use.")
print("\nNext steps:")
print("1. Train the model: python train_deep_learning_control.py")
print("2. Use the controller: tinyphysics.py --controller deep_learning_control")
print("3. Enable online learning: Set enable_online_learning=True in controller")
