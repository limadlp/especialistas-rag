"""Local text processing and the shared multilingual embedding model."""
from functools import lru_cache
import os
import re
import unicodedata

from . import settings

STOP = set("a o as os um uma uns umas de da do das dos em no na nos nas e ou que qual quais como para por com se ao aos à às é são foi ser sobre segundo explique livro skinner comportamento humano ciência ciencia autor texto ele ela isso esse essa uma pode".split())


def lexical(text: str) -> list[str]:
    text = "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))
    return [w for w in re.findall(r"\w+", text) if len(w) > 1 and w not in STOP]


@lru_cache(maxsize=1)
def load_encoder():
    # A missing model raises locally; ordinary execution never downloads weights.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    torch.set_num_threads(2)
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(str(settings.EMBEDDING_MODEL), local_files_only=True, device="cpu")
