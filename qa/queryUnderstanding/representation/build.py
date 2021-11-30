from fvcore.common.registry import Registry

REPRESENTATION_REGISTRY = Registry("REPRESENTATION")  # noqa F401 isort:skip
REPRESENTATION_REGISTRY.__doc__ = """
Registry for semantic retrieval, i.e. the word2vec model.
The registered object will be called with `obj(cfg)`
and expected to return a `Embedding` object.
"""


def build_embedding(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    embedding = cfg.MODEL.EMBEDDING
    embedding = REPRESENTATION_REGISTRY.get(embedding)(cfg)
    return embedding