from .posenet import PoseNetPlus
from .atloc import AtLoc
from .transposenet import TransPoseNet
from .ms_transformer import MSTransformer

MODEL_REGISTRY = {
    "posenet": PoseNetPlus,
    "atloc": AtLoc,
    "transposenet": TransPoseNet,
    "ms_transformer": MSTransformer,
}


def build_model(cfg):
    name = cfg["model"]["name"]
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    model_cls = MODEL_REGISTRY[name]
    return model_cls(**cfg["model"].get("params", {}))
