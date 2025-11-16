import hashlib
from typing import Union

import pandas as pd

from .config import SEED_DIGEST_SIZE, SEED_HASH_COLUMNS, SEED_MODULO


def compute_segment_seed(data: pd.DataFrame, identifier: Union[str, bytes] = b"") -> int:
  """
  Produce a reproducible seed for a driving segment using the processed dataframe
  and an optional identifier (usually the filename).
  """
  missing_cols = set(SEED_HASH_COLUMNS) - set(data.columns)
  if missing_cols:
    raise ValueError(f"Missing columns for seed computation: {missing_cols}")

  hashed_series = pd.util.hash_pandas_object(data[list(SEED_HASH_COLUMNS)], index=True)
  payload = hashed_series.to_numpy().tobytes()
  if isinstance(identifier, str):
    identifier = identifier.encode('utf-8')
  digest = hashlib.blake2b(payload + identifier, digest_size=SEED_DIGEST_SIZE).digest()
  return int.from_bytes(digest, byteorder='big', signed=False) % SEED_MODULO
