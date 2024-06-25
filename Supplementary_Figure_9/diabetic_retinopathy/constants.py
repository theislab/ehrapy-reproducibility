from enum import Enum
from pathlib import Path
import numpy as np


ROOT = Path(__file__).parent.resolve().parent

PROJECT_DIR = ROOT / "project_folder"
DATA_DIR = PROJECT_DIR / "data"
RAWDATA_DIR = DATA_DIR / "raw"

DATASPLIT_DIR = DATA_DIR / "data_split"
MODELS_DIR = PROJECT_DIR / "experiments"
RESULTS_DIR = PROJECT_DIR / "results"
WEIGHTS_DIR = PROJECT_DIR / "model_weights"
PROCESSED_DIR = DATA_DIR / "processed"
TRAIN_CONFIG = PROJECT_DIR / "configs" / "train_configs"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"

# constants for phases
class PhaseType(Enum):
    train = "train"
    val = "val"
    test = "test"


label_map = {0: "Normal", 1: "Mild", 2: "Medium", 3: "Severe", 4: "Proliferative"}
