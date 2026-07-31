import os

# Force offline mode – never download from Hugging Face Hub
os.environ["HF_HUB_OFFLINE"] = "1"

from .kronos import KronosTokenizer, Kronos, KronosPredictor

# ------------------------------------------------------------
# Local weights directory
# ------------------------------------------------------------
_WEIGHTS_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")

WEIGHTS = {
    "Kronos-Tokenizer-base": os.path.join(_WEIGHTS_ROOT, "Kronos-Tokenizer-base"),
    "Kronos-Tokenizer-2k":   os.path.join(_WEIGHTS_ROOT, "Kronos-Tokenizer-2k"),
    "Kronos-mini":           os.path.join(_WEIGHTS_ROOT, "Kronos-mini"),
    "Kronos-small":          os.path.join(_WEIGHTS_ROOT, "Kronos-small"),
    "Kronos-base":           os.path.join(_WEIGHTS_ROOT, "Kronos-base"),
}


def load_tokenizer(model_name: str = "Kronos-Tokenizer-base"):
    """Load a KronosTokenizer from local weights.

    Parameters
    ----------
    model_name : str
        One of ``WEIGHTS`` keys (e.g. ``"Kronos-Tokenizer-base"``).

    Returns
    -------
    KronosTokenizer
    """
    path = WEIGHTS.get(model_name)
    if path is None:
        raise ValueError(
            f"Unknown tokenizer '{model_name}'. Available: {list(WEIGHTS.keys())}"
        )
    return KronosTokenizer.from_pretrained(path)


def load_model(model_name: str = "Kronos-base"):
    """Load a Kronos model from local weights.

    Parameters
    ----------
    model_name : str
        One of ``WEIGHTS`` keys (e.g. ``"Kronos-base"``).

    Returns
    -------
    Kronos
    """
    path = WEIGHTS.get(model_name)
    if path is None:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(WEIGHTS.keys())}"
        )
    return Kronos.from_pretrained(path)


model_dict = {
    'kronos_tokenizer': KronosTokenizer,
    'kronos': Kronos,
    'kronos_predictor': KronosPredictor
}


def get_model_class(model_name):
    if model_name in model_dict:
        return model_dict[model_name]
    else:
        print(f"Model {model_name} not found in model_dict")
        raise NotImplementedError
