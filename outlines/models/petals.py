from typing import Optional
from outlines.transformers import TransformersTokenizer, Transformers

__all__ = ["petals"]

def petals(model_name: str, device: Optional[str] = None, **model_kwargs):
    try:
        from petals import AutoDistributedModelForCausalLM
    except ImportError:
        raise ImportError(
            "The `petals` library needs to be installed in order to use `petals` models, e.g. pip install git+https://github.com/bigscience-workshop/petals."
        )

    model = AutoDistributedModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    tokenizer = TransformersTokenizer(model_name)

    return Transformers(model, tokenizer, device)
