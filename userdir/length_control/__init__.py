from fairseq.models import register_model_architecture
from fairseq.models.transformer import base_architecture
from . import lc_transformer, lc_translation

@register_model_architecture('lc_transformer', 'lc_transformer')
def lc_transformer_arch(args):
    base_architecture(args)
