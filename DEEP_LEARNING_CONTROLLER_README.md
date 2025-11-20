# Deep Learning Controller

An unsupervised neural network controller with convolutional layers for lateral vehicle control.

## Overview

This controller uses a deep convolutional neural network that:
- **Input**: Takes historical time-series data and future trajectory plans
- **Output**: Produces both control actions AND predictions of next lateral acceleration
- **Learning**: Uses actual lat_accel from simulator as supervision signal (unsupervised learning)

### Key Features

1. **Convolutional Architecture**: Uses Conv1D layers to process time-series history
2. **Dual Output Heads**:
   - Control action (steering command)
   - Predicted next lateral acceleration
3. **Unsupervised Learning**: Loss is computed from prediction error of actual lat_accel
4. **Online Learning**: Can continue learning during inference (optional)

## Architecture

### Network Structure

```
History Input (20 timesteps × 7 channels):
  - error (target - current lataccel)
  - current_lataccel
  - target_lataccel
  - v_ego (vehicle velocity)
  - a_ego (vehicle acceleration)
  - roll_lataccel
  - action (previous control action)

Future Plan Input (10 timesteps × 4 features):
  - lataccel
  - roll_lataccel
  - v_ego
  - a_ego

Conv1D Layers (16 → 32 → 64 filters)
  ↓
Flatten + Concatenate with Future Plan
  ↓
Fully Connected Layers (128 → 64)
  ↓
Dual Output Heads:
  - Action Head (tanh activation) → control action
  - Prediction Head (linear) → predicted next lataccel
```

### Loss Function

```
total_loss = prediction_loss + regularization
where:
  prediction_loss = (predicted_lataccel - actual_lataccel)²
  regularization = 0.01 × action²
```

The network learns by minimizing the error between its predicted lateral acceleration and the actual lat_accel produced by the simulator.

## Files

- **[deep_learning_network.py](deep_learning_network.py)**: Neural network implementation
  - `DeepLearningControlNetwork`: Main network class with conv layers
  - `ExperienceBuffer`: Buffer for storing training experiences

- **[controllers/deep_learning_control.py](controllers/deep_learning_control.py)**: Controller implementation
  - Integrates network into simulator framework
  - Handles history management and data preparation
  - Supports online learning via `on_simulation_update` hook

- **[train_deep_learning_control.py](train_deep_learning_control.py)**: Training script
  - Online learning from simulator episodes
  - Experience replay for stability
  - Periodic model checkpointing

## Usage

### 1. Training

Train the network on simulation data:

```bash
python train_deep_learning_control.py
```

Training parameters:
- Episodes: 50 (default)
- History length: 20 timesteps
- Future plan length: 10 timesteps
- Learning rate: 0.0001
- Batch size: 32
- Buffer capacity: 5000

The trained model will be saved to `models/deep_learning_control.pkl`.

### 2. Inference

Use the trained controller:

```bash
python tinyphysics.py --controller deep_learning_control
```

### 3. Online Learning (Optional)

To enable continued learning during inference, modify the controller instantiation:

```python
from controllers.deep_learning_control import Controller

controller = Controller(
  model_path="models/deep_learning_control.pkl",
  enable_online_learning=True,
  online_learning_rate=0.00001  # Lower rate for stability
)
```

## How It Works

### Training Phase

1. **Episode Initialization**: Load segment data from simulator
2. **Warmup**: Use simple proportional control for first 10 steps to build initial history
3. **Network Control**:
   - Prepare history matrix from recent observations
   - Prepare future plan matrix from trajectory
   - Network predicts action and next lat_accel
4. **Simulation Step**: Apply action, get actual lat_accel from simulator
5. **Learning Update**:
   - Compute loss from prediction error
   - Backpropagate gradients
   - Update network weights
6. **Experience Replay**: Periodically train on samples from buffer for stability

### Inference Phase

1. **History Tracking**: Maintain rolling buffers of recent states and actions
2. **Input Preparation**: Convert histories to normalized network inputs
3. **Prediction**: Network outputs control action (and lat_accel prediction)
4. **Optional Online Learning**: If enabled, update network when actual lat_accel arrives via `on_simulation_update` hook

### Key Innovation: Unsupervised Learning

Traditional controllers require manual tuning or labeled training data. This controller learns purely from:
- The vehicle dynamics (embedded in simulator)
- The prediction error of lateral acceleration

By trying to accurately predict what the simulator will do next, the network implicitly learns optimal control strategies.

## Testing

Run the test suite:

```bash
python test_deep_learning_controller.py
```

Tests verify:
- Controller import and instantiation
- Reset functionality
- Action computation
- Diagnostics
- Network forward/backward passes
- Model save/load

## Customization

### Network Architecture

Modify in [deep_learning_network.py](deep_learning_network.py):

```python
network = DeepLearningControlNetwork(
  history_length=20,        # Longer history = more context
  future_length=10,         # Future lookahead
  num_channels=7,           # Input features per timestep
  conv_filters=(16, 32, 64),  # Conv layer sizes
  kernel_size=3,            # Conv kernel width
  fc_dims=(128, 64),        # Fully connected layers
  learning_rate=0.0001      # Learning rate
)
```

### Training Parameters

Modify in [train_deep_learning_control.py](train_deep_learning_control.py):

```python
trainer = DeepLearningControlTrainer(
  network=network,
  batch_size=32,           # Replay batch size
  buffer_capacity=5000     # Experience buffer size
)

trainer.train(
  num_episodes=50,         # Training episodes
  save_interval=5          # Save every N episodes
)
```

## Advantages

1. **No Manual Tuning**: Learns control strategy from simulator feedback
2. **Adaptive**: Can continue learning online as conditions change
3. **Context-Aware**: Conv layers extract temporal patterns from history
4. **Predictive**: Learns vehicle dynamics through lat_accel prediction task

## Limitations

1. **Computational Cost**: More expensive than PID controllers
2. **Training Required**: Needs simulation episodes to learn (vs. analytical PID)
3. **Black Box**: Less interpretable than traditional controllers
4. **Simplified Backprop**: Current implementation uses approximate gradients through conv layers (full implementation would benefit from PyTorch/TensorFlow)

## Future Improvements

1. **Full Conv Backprop**: Use proper framework for exact gradients
2. **Recurrent Layers**: Add LSTM/GRU for better temporal modeling
3. **Multi-Task Learning**: Add auxiliary prediction tasks (velocity, acceleration)
4. **Transfer Learning**: Pre-train on multiple vehicle types
5. **Attention Mechanism**: Learn which parts of history/future are most important

## Diagnostics

The controller provides the following diagnostic metrics:

```python
diagnostics = controller.get_diagnostics()
# Returns:
# {
#   'avg_prediction_error': float,      # Average lat_accel prediction error
#   'control_steps': int,                # Total control steps
#   'online_learning_enabled': bool,     # Whether online learning is active
#   'network_train_steps': int          # Total network updates
# }
```

## References

This approach is inspired by:
- Model-based reinforcement learning (MBRL)
- Self-supervised learning from prediction
- End-to-end learning for autonomous driving
