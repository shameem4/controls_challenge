from abc import ABC, abstractmethod
from typing import List

import numpy as np
import onnxruntime as ort

from .config import (
  LATACCEL_RANGE,
  MODEL_SAMPLE_TEMPERATURE,
  ORT_INTER_OP_THREADS,
  ORT_INTRA_OP_THREADS,
  ORT_LOG_SEVERITY_LEVEL,
  VOCAB_SIZE,
)
from .types import State


class LataccelTokenizer:
  def __init__(self):
    self.vocab_size = VOCAB_SIZE
    self.bins = np.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], self.vocab_size)

  def encode(self, value):
    value = self.clip(value)
    return np.digitize(value, self.bins, right=True)

  def decode(self, token):
    return self.bins[token]

  def clip(self, value):
    return np.clip(value, LATACCEL_RANGE[0], LATACCEL_RANGE[1])


class BasePhysicsModel(ABC):
  """Interface for any plant model that can predict lateral acceleration."""

  @abstractmethod
  def predict_lataccel(self, sim_states: List[State], actions: List[float], past_preds: List[float]) -> float:
    """Return the next lateral acceleration estimate."""

  def seed(self, seed: int) -> None:
    """Optional hook to seed internal RNG state."""
    return None


class TinyPhysicsModel(BasePhysicsModel):
  """Wrapper around the ONNX-based simulator that predicts lateral acceleration."""

  def __init__(self, model_path: str, debug: bool = False) -> None:
    del debug  # retained for backward compatibility; we always run in CPU mode
    self.tokenizer = LataccelTokenizer()
    options = ort.SessionOptions()
    options.intra_op_num_threads = ORT_INTRA_OP_THREADS
    options.inter_op_num_threads = ORT_INTER_OP_THREADS
    options.log_severity_level = ORT_LOG_SEVERITY_LEVEL
    provider = 'CPUExecutionProvider'

    with open(model_path, "rb") as f:
      self.ort_session = ort.InferenceSession(f.read(), options, [provider])
    self._rng = np.random.default_rng()

  def seed(self, seed: int) -> None:
    self._rng = np.random.default_rng(seed)

  def _softmax(self, x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

  def _predict_token(self, input_data: dict, temperature=1.) -> int:
    res = self.ort_session.run(None, input_data)[0]
    probs = self._softmax(res / temperature, axis=-1)
    # we only care about the last timestep (batch size is just 1)
    assert probs.shape[0] == 1
    assert probs.shape[2] == VOCAB_SIZE
    return self._rng.choice(probs.shape[2], p=probs[0, -1])

  def predict_lataccel(self, sim_states: List[State], actions: List[float], past_preds: List[float]) -> float:
    tokenized_actions = self.tokenizer.encode(past_preds)
    raw_states = [[state.roll_lataccel, state.v_ego, state.a_ego] for state in sim_states]
    states = np.column_stack([actions, raw_states])
    input_data = {
      'states': np.expand_dims(states, axis=0).astype(np.float32),
      'tokens': np.expand_dims(tokenized_actions, axis=0).astype(np.int64)
    }
    return self.tokenizer.decode(self._predict_token(input_data, temperature=MODEL_SAMPLE_TEMPERATURE))
