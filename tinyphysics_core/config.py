from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

ACC_G = 9.81
FPS = 10
CONTROL_START_IDX = 100
COST_END_IDX = 500
CONTEXT_LENGTH = 20
VOCAB_SIZE = 1024
LATACCEL_RANGE = (-5.0, 5.0)
STEER_RANGE = (-2.0, 2.0)
MAX_ACC_DELTA = 0.5
DEL_T = 0.1
LAT_ACCEL_COST_MULTIPLIER = 50.0
FUTURE_PLAN_SECONDS = 5
FUTURE_PLAN_STEPS = FPS * FUTURE_PLAN_SECONDS

MODEL_SAMPLE_TEMPERATURE = 0.8
ORT_INTRA_OP_THREADS = 1
ORT_INTER_OP_THREADS = 1
ORT_LOG_SEVERITY_LEVEL = 3

HISTOGRAM_BINS = tuple(range(0, 1000, 10))

DATASET_URL = "https://huggingface.co/datasets/commaai/commaSteeringControl/resolve/main/data/SYNTHETIC_V0.zip"
DATASET_PATH = BASE_DIR / "data"
MODELS_PATH = BASE_DIR / "models"
DEFAULT_MODEL_PATH = MODELS_PATH / "tinyphysics.onnx"

SEED_HASH_COLUMNS = ('roll_lataccel', 'v_ego', 'a_ego', 'target_lataccel', 'steer_command')
SEED_DIGEST_SIZE = 16
SEED_MODULO = 2**32
