from .models import MACE, RACE, RACEUnified

MODEL_REGISTRY = {
    "race": RACE,
    "mace": MACE,
    "race_multihead": RACEUnified,
    "race_unified": RACEUnified,
}

__all__ = ["RACE", "MACE", "RACEUnified", "MODEL_REGISTRY"]
