# Adaptive PID Gain Network

This system implements a self-supervised neural network for adaptive PID gain tuning. The network learns to predict optimal PID gains based on the current driving context, road characteristics, and historical performance.

## Overview

The system consists of three main components:

1. **Adaptive Controller** (`controllers/pid_adaptive.py`): PID controller with neural network-based gain adaptation
2. **Gain Network** (`gain_adapter_network.py`): Simple feedforward neural network that predicts PID gains
3. **Training Script** (`train_adaptive_gains.py`): Self-supervised learning to optimize gains

## How It Works

### Feature Extraction

The controller extracts a rich feature vector (48 dimensions) including:

- **Current State** (8 features):
  - Error (target - current lataccel)
  - Current lateral acceleration
  - Target lateral acceleration
  - Vehicle velocity (v_ego)
  - Longitudinal acceleration (a_ego)
  - Roll-induced lateral acceleration
  - Integrator state
  - Previous error

- **Historical Context** (30 features):
  - Last 10 error values
  - Last 10 lateral acceleration values
  - Last 10 velocity values

- **Future Plan** (8 features):
  - Next 4 target lateral accelerations
  - Next 4 velocities

- **Road Characteristics** (2 features):
  - Variance of upcoming lateral accelerations
  - Maximum upcoming lateral acceleration

### Neural Network Architecture

```
Input (48) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(4, Tanh)
```

The network outputs 4 gain adjustments:
- Proportional gain (P)
- Integral gain (I)
- Derivative gain (D)
- Feedforward gain

These are normalized and then mapped to valid gain ranges:
- P: [0.1, 0.8]
- I: [0.0, 0.2]
- D: [-0.3, 0.0]
- FF: [0.3, 1.2]

### Self-Supervised Learning

The training process is fully self-supervised and doesn't require labeled data:

1. **Exploration Phase**: Controller runs with random gain combinations or network predictions
2. **Experience Collection**: Each step records (features, gains, cost)
3. **Replay Buffer**: Experiences are stored with their associated costs
4. **Imitation Learning**: Network learns to imitate the gains that produced the lowest costs

Key training features:
- **Prioritized Experience Replay**: 70% of training samples come from best-performing experiences
- **Exploration Decay**: Gradually shifts from random exploration to network exploitation
- **Gain Statistics Adaptation**: Network normalizes gains based on observed best performances

## Usage

### 1. Training the Network

Basic training:
```bash
python train_adaptive_gains.py
```

Quick training on worst segments:
```bash
# Train on first 10 worst segments
python quick_train_adaptive.py --num_segments 10

# Train on all 26 worst segments
python quick_train_adaptive.py
```

Advanced options:
```bash
python train_adaptive_gains.py \
  --data_path data/training_segments \
  --num_epochs 50 \
  --episodes_per_epoch 20 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --segment_limit 100 \
  --output_path models/gain_adapter.pkl \
  --no-resume  # Start fresh, don't load previous best checkpoint
```

Parameters:
- `--num_epochs`: Number of training epochs (default: 50)
- `--episodes_per_epoch`: Segments to run per epoch (default: 20)
- `--batch_size`: Training batch size (default: 32)
- `--learning_rate`: Learning rate (default: 0.001)
- `--segment_limit`: Maximum segments to use from dataset (default: 100)
- `--output_path`: Where to save the trained model
- `--no-resume`: Don't resume from best checkpoint (default: False, will resume)

The script will:
- **Automatically resume from best checkpoint** (unless `--no-resume` is specified)
- Explore different gain combinations
- Collect experiences in a replay buffer
- Train the network to predict good gains
- Evaluate every 5 epochs
- **Save best checkpoint whenever a new best avg cost is achieved**
- Save periodic checkpoints every 10 epochs
- Save final model and training history to JSON

### 2. Evaluating the Network

Compare adaptive vs baseline:
```bash
python eval_adaptive.py --compare --num_segments 20
```

Evaluate specific segments:
```bash
python eval_adaptive.py --segments 00000 00006 00012 00014
```

Evaluate all segments:
```bash
python eval_adaptive.py
```

Parameters:
- `--data_path`: Dataset directory (default: data/)
- `--network_path`: Path to trained network (default: models/gain_adapter.pkl)
- `--num_segments`: Number of segments to evaluate
- `--baseline`: Evaluate baseline controller only
- `--compare`: Compare baseline and adaptive controllers
- `--segments`: Specific segment IDs to evaluate

### 3. Using in Custom Code

```python
from gain_adapter_network import GainAdapterNetwork
from controllers.pid_adaptive import Controller

# Load trained network
network = GainAdapterNetwork.load("models/gain_adapter.pkl")

# Create controller with network
controller = Controller()
controller.use_adaptive_gains = True
controller.gain_adapter = network

# Use in simulation
action = controller.update(target_lataccel, current_lataccel, state, future_plan, control_state)
```

## Tunable Parameters

All controller parameters are exposed and tunable:

### PID Gains (adapted by network)
- `p`: Proportional gain (default: 0.3)
- `i`: Integral gain (default: 0.07)
- `d`: Derivative gain (default: -0.1)

### Feedforward Parameters
- `steer_factor`: Steering sensitivity (default: 13.0)
- `steer_sat_v`: Velocity for steering saturation (default: 20.0)
- `steer_command_sat`: Maximum steering command (default: 2.0)
- `feedforward_gain`: Feedforward multiplier (default: 0.8, adapted by network)

### Future Blending
- `future_weights`: Weights for blending future targets (default: [5, 6, 7, 8])

### PID Scaling
- `pid_target_scale_threshold`: Threshold for PID scaling (default: 1.0)
- `pid_target_scale_rate`: Rate of PID reduction at high targets (default: 0.23)
- `longitudinal_gain_scale`: Gain reduction based on longitudinal accel (default: 10.0)

### Neural Network
- `use_adaptive_gains`: Enable/disable network adaptation (default: True)
- `history_length`: Number of past steps to include in features (default: 10)

## Training Strategy

### Self-Supervised Learning Approach

The system uses a novel self-supervised approach:

1. **No Ground Truth Required**: Unlike supervised learning, we don't need labeled "optimal gains"
2. **Cost-Based Learning**: The network learns from the relationship between (state, gains) → cost
3. **Exploration-Exploitation**: Balances trying new gains vs using learned predictions
4. **Best-Practice Imitation**: Preferentially learns from experiences with lowest cost

### Why This Works

- **Direct Optimization**: Network learns gains that minimize actual cost function
- **Context-Aware**: Adapts gains based on velocity, road curvature, historical errors
- **Distributed Learning**: Each segment contributes experiences across diverse scenarios
- **No Overfitting to Labels**: Learns from actual performance rather than human intuition

## Expected Performance

Based on the training approach, you should expect:

1. **Initial Epochs (1-10)**:
   - High exploration, random performance
   - Building up experience buffer
   - Network learning basic patterns

2. **Mid Training (10-30)**:
   - Decreasing exploration rate
   - Network starts outperforming random gains
   - Gain statistics stabilize

3. **Late Training (30-50)**:
   - Minimal exploration (5%)
   - Consistent performance improvement
   - Network fine-tuning based on best experiences

## Troubleshooting

### Network not improving
- Increase `episodes_per_epoch` to collect more data
- Decrease `learning_rate` for more stable learning
- Increase `best_sample_ratio` to learn more from good examples

### High variance in results
- Increase `segment_limit` to train on more diverse scenarios
- Adjust `exploration_decay` to explore more
- Check gain ranges are appropriate for your vehicle

### Evaluation crashes
- Ensure `pid_adaptive` controller is importable
- Check that network file exists at specified path
- Verify dataset path contains valid CSV files

## Architecture Decisions

### Why This Architecture?

1. **Simple Network**: 2 hidden layers are sufficient for this regression task
2. **ReLU Hidden Layers**: Standard for regression, good gradient flow
3. **Tanh Output**: Bounds gain adjustments to [-1, 1] range
4. **No Convolutional Layers**: Time series is short, fully connected is adequate
5. **Manual Feature Engineering**: More interpretable than end-to-end learning

### Alternatives Considered

- **Reinforcement Learning**: More complex, requires reward shaping
- **Recurrent Networks**: Overkill for 10-step history
- **Direct Cost Prediction**: Doesn't provide actionable gains
- **Supervised Learning**: Requires labeled optimal gains (unavailable)

## Future Improvements

1. **Online Learning**: Update network during deployment
2. **Multi-Objective Optimization**: Separate networks for lataccel vs jerk
3. **Ensemble Methods**: Combine multiple networks
4. **Meta-Learning**: Fast adaptation to new vehicles
5. **Attention Mechanisms**: Better use of historical context
6. **Uncertainty Estimation**: Confidence-based exploration

## References

- Self-supervised learning for control: Learning by doing
- Imitation learning: Learn from best experiences
- Experience replay: Stabilize training with diverse samples
- Exploration-exploitation: Balance discovery and optimization
