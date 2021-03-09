# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.modules import (
    AdaptiveSoftmax,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from torch import Tensor
from fairseq.models.transformer import (
    TransformerModel,
    TransformerDecoder,
    Embedding,
    base_architecture
)

from .lc_transformer_layer import LCTransformerDecoderLayer
from .lc_positional_embedding import LCPositionalEmbedding

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model("lc_transformer")
class LengthControlTransformerModel(TransformerModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--decoder-output-dim', type=int, metavar='N',
                            help='decoder output dimension (extra linear layer '
                                 'if different from decoder embed dim')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--layernorm-embedding', action='store_true',
                            help='add layernorm to embedding')
        parser.add_argument('--no-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')
        # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
        parser.add_argument('--no-cross-attention', default=False, action='store_true',
                            help='do not perform cross-attention')
        parser.add_argument('--cross-self-attention', default=False, action='store_true',
                            help='perform cross+self-attention')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for decoder')
        parser.add_argument('--encoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--decoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
        parser.add_argument('--quant-noise-pq', type=float, metavar='D', default=0,
                            help='iterative PQ quantization noise at training time')
        parser.add_argument('--quant-noise-pq-block-size', type=int, metavar='D', default=8,
                            help='block size of quantization noise at training time')
        parser.add_argument('--quant-noise-scalar', type=float, metavar='D', default=0,
                            help='scalar quantization noise and scalar quantization at training time')
        parser.add_argument('--represent-length-by-lrpe', default=False, action='store_true',
                            help='represent target length by length ratio positional encoding')
        parser.add_argument('--represent-length-by-ldpe', default=False, action='store_true',
                            help='represent target length by length difference positional encoding')
        parser.add_argument('--ordinary-sinpos', default=False, action='store_true',
                            help='use ordinary sinusoidal positional encoding (absolute position)')
        # fmt: on

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.args = args
        self.supports_align_args = True

    def forward(
        self,
        src_tokens,
        src_lengths,
        tgt_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            tgt_lengths=tgt_lengths,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out

    # @classmethod
    # def build_model(cls, args, task):
    #     """Build a new model instance."""
    #
    #     # make sure all arguments are present in older models
    #     base_architecture(args)
    #
    #     if args.encoder_layers_to_keep:
    #         args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
    #     if args.decoder_layers_to_keep:
    #         args.decoder_layers = len(args.decoder_layers_to_keep.split(","))
    #
    #     if getattr(args, "max_source_positions", None) is None:
    #         args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
    #     if getattr(args, "max_target_positions", None) is None:
    #         args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS
    #
    #     src_dict, tgt_dict = task.source_dictionary, task.target_dictionary
    #
    #     if args.share_all_embeddings:
    #         if src_dict != tgt_dict:
    #             raise ValueError("--share-all-embeddings requires a joined dictionary")
    #         if args.encoder_embed_dim != args.decoder_embed_dim:
    #             raise ValueError(
    #                 "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
    #             )
    #         if args.decoder_embed_path and (
    #             args.decoder_embed_path != args.encoder_embed_path
    #         ):
    #             raise ValueError(
    #                 "--share-all-embeddings not compatible with --decoder-embed-path"
    #             )
    #         encoder_embed_tokens = cls.build_embedding(
    #             args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
    #         )
    #         decoder_embed_tokens = encoder_embed_tokens
    #         args.share_decoder_input_output_embed = True
    #     else:
    #         encoder_embed_tokens = cls.build_embedding(
    #             args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
    #         )
    #         decoder_embed_tokens = cls.build_embedding(
    #             args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
    #         )
    #
    #     encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
    #     decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
    #     return cls(args, encoder, decoder)

    # @classmethod
    # def build_embedding(cls, args, dictionary, embed_dim, path=None):
    #     num_embeddings = len(dictionary)
    #     padding_idx = dictionary.pad()
    #
    #     emb = Embedding(num_embeddings, embed_dim, padding_idx)
    #     # if provided, load from preloaded dictionaries
    #     if path:
    #         embed_dict = utils.parse_embedding(path)
    #         utils.load_embedding(embed_dict, dictionary, emb)
    #     return emb

    # @classmethod
    # def build_encoder(cls, args, src_dict, embed_tokens):
    #     return TransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return LCTransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )


# class TransformerEncoder(FairseqEncoder):
#     """
#     Transformer encoder consisting of *args.encoder_layers* layers. Each layer
#     is a :class:`TransformerEncoderLayer`.
#
#     Args:
#         args (argparse.Namespace): parsed command-line arguments
#         dictionary (~fairseq.data.Dictionary): encoding dictionary
#         embed_tokens (torch.nn.Embedding): input embedding
#     """
#
#     def __init__(self, args, dictionary, embed_tokens):
#         super().__init__(dictionary)
#         self.register_buffer("version", torch.Tensor([3]))
#
#         self.dropout_module = FairseqDropout(
#             args.dropout, module_name=self.__class__.__name__
#         )
#         self.encoder_layerdrop = args.encoder_layerdrop
#
#         embed_dim = embed_tokens.embedding_dim
#         self.padding_idx = embed_tokens.padding_idx
#         self.max_source_positions = args.max_source_positions
#
#         self.embed_tokens = embed_tokens
#
#         self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)
#
#         self.embed_positions = (
#             PositionalEmbedding(
#                 args.max_source_positions,
#                 embed_dim,
#                 self.padding_idx,
#                 learned=args.encoder_learned_pos,
#             )
#             if not args.no_token_positional_embeddings
#             else None
#         )
#
#         if getattr(args, "layernorm_embedding", False):
#             self.layernorm_embedding = LayerNorm(embed_dim)
#         else:
#             self.layernorm_embedding = None
#
#         if not args.adaptive_input and args.quant_noise_pq > 0:
#             self.quant_noise = apply_quant_noise_(
#                 nn.Linear(embed_dim, embed_dim, bias=False),
#                 args.quant_noise_pq,
#                 args.quant_noise_pq_block_size,
#             )
#         else:
#             self.quant_noise = None
#
#         if self.encoder_layerdrop > 0.0:
#             self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
#         else:
#             self.layers = nn.ModuleList([])
#         self.layers.extend(
#             [self.build_encoder_layer(args) for i in range(args.encoder_layers)]
#         )
#         self.num_layers = len(self.layers)
#
#         if args.encoder_normalize_before:
#             self.layer_norm = LayerNorm(embed_dim)
#         else:
#             self.layer_norm = None
#
#     def build_encoder_layer(self, args):
#         return TransformerEncoderLayer(args)
#
#     def forward_embedding(
#         self, src_tokens, token_embedding: Optional[torch.Tensor] = None
#     ):
#         # embed tokens and positions
#         if token_embedding is None:
#             token_embedding = self.embed_tokens(src_tokens)
#         x = embed = self.embed_scale * token_embedding
#         if self.embed_positions is not None:
#             x = embed + self.embed_positions(src_tokens)
#         if self.layernorm_embedding is not None:
#             x = self.layernorm_embedding(x)
#         x = self.dropout_module(x)
#         if self.quant_noise is not None:
#             x = self.quant_noise(x)
#         return x, embed
#
#     def forward(
#         self,
#         src_tokens,
#         src_lengths,
#         return_all_hiddens: bool = False,
#         token_embeddings: Optional[torch.Tensor] = None,
#     ):
#         """
#         Args:
#             src_tokens (LongTensor): tokens in the source language of shape
#                 `(batch, src_len)`
#             src_lengths (torch.LongTensor): lengths of each source sentence of
#                 shape `(batch)`
#             return_all_hiddens (bool, optional): also return all of the
#                 intermediate hidden states (default: False).
#             token_embeddings (torch.Tensor, optional): precomputed embeddings
#                 default `None` will recompute embeddings
#
#         Returns:
#             namedtuple:
#                 - **encoder_out** (Tensor): the last encoder layer's output of
#                   shape `(src_len, batch, embed_dim)`
#                 - **encoder_padding_mask** (ByteTensor): the positions of
#                   padding elements of shape `(batch, src_len)`
#                 - **encoder_embedding** (Tensor): the (scaled) embedding lookup
#                   of shape `(batch, src_len, embed_dim)`
#                 - **encoder_states** (List[Tensor]): all intermediate
#                   hidden states of shape `(src_len, batch, embed_dim)`.
#                   Only populated if *return_all_hiddens* is True.
#         """
#         x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)
#
#         # B x T x C -> T x B x C
#         x = x.transpose(0, 1)
#
#         # compute padding mask
#         encoder_padding_mask = src_tokens.eq(self.padding_idx)
#
#         encoder_states = [] if return_all_hiddens else None
#
#         # encoder layers
#         for layer in self.layers:
#             x = layer(x, encoder_padding_mask)
#             if return_all_hiddens:
#                 assert encoder_states is not None
#                 encoder_states.append(x)
#
#         if self.layer_norm is not None:
#             x = self.layer_norm(x)
#
#         return EncoderOut(
#             encoder_out=x,  # T x B x C
#             encoder_padding_mask=encoder_padding_mask,  # B x T
#             encoder_embedding=encoder_embedding,  # B x T x C
#             encoder_states=encoder_states,  # List[T x B x C]
#             src_tokens=None,
#             src_lengths=None,
#         )
#
#     @torch.jit.export
#     def reorder_encoder_out(self, encoder_out: EncoderOut, new_order):
#         """
#         Reorder encoder output according to *new_order*.
#
#         Args:
#             encoder_out: output from the ``forward()`` method
#             new_order (LongTensor): desired order
#
#         Returns:
#             *encoder_out* rearranged according to *new_order*
#         """
#         """
#         Since encoder_padding_mask and encoder_embedding are both of type
#         Optional[Tensor] in EncoderOut, they need to be copied as local
#         variables for Torchscript Optional refinement
#         """
#         encoder_padding_mask: Optional[Tensor] = encoder_out.encoder_padding_mask
#         encoder_embedding: Optional[Tensor] = encoder_out.encoder_embedding
#
#         new_encoder_out = (
#             encoder_out.encoder_out
#             if encoder_out.encoder_out is None
#             else encoder_out.encoder_out.index_select(1, new_order)
#         )
#         new_encoder_padding_mask = (
#             encoder_padding_mask
#             if encoder_padding_mask is None
#             else encoder_padding_mask.index_select(0, new_order)
#         )
#         new_encoder_embedding = (
#             encoder_embedding
#             if encoder_embedding is None
#             else encoder_embedding.index_select(0, new_order)
#         )
#         src_tokens = encoder_out.src_tokens
#         if src_tokens is not None:
#             src_tokens = src_tokens.index_select(0, new_order)
#
#         src_lengths = encoder_out.src_lengths
#         if src_lengths is not None:
#             src_lengths = src_lengths.index_select(0, new_order)
#
#         encoder_states = encoder_out.encoder_states
#         if encoder_states is not None:
#             for idx, state in enumerate(encoder_states):
#                 encoder_states[idx] = state.index_select(1, new_order)
#
#         return EncoderOut(
#             encoder_out=new_encoder_out,  # T x B x C
#             encoder_padding_mask=new_encoder_padding_mask,  # B x T
#             encoder_embedding=new_encoder_embedding,  # B x T x C
#             encoder_states=encoder_states,  # List[T x B x C]
#             src_tokens=src_tokens,  # B x T
#             src_lengths=src_lengths,  # B x 1
#         )
#
#     def max_positions(self):
#         """Maximum input length supported by the encoder."""
#         if self.embed_positions is None:
#             return self.max_source_positions
#         return min(self.max_source_positions, self.embed_positions.max_positions)
#
#     def upgrade_state_dict_named(self, state_dict, name):
#         """Upgrade a (possibly old) state dict for new versions of fairseq."""
#         if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
#             weights_key = "{}.embed_positions.weights".format(name)
#             if weights_key in state_dict:
#                 print("deleting {0}".format(weights_key))
#                 del state_dict[weights_key]
#             state_dict[
#                 "{}.embed_positions._float_tensor".format(name)
#             ] = torch.FloatTensor(1)
#         for i in range(self.num_layers):
#             # update layer norms
#             self.layers[i].upgrade_state_dict_named(
#                 state_dict, "{}.layers.{}".format(name, i)
#             )
#
#         version_key = "{}.version".format(name)
#         if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
#             # earlier checkpoints did not normalize after the stack of layers
#             self.layer_norm = None
#             self.normalize = False
#             state_dict[version_key] = torch.Tensor([1])
#         return state_dict


class LCTransformerDecoder(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        self.args = args
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.decoder_layerdrop = args.decoder_layerdrop
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = args.decoder_output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        self.ordinary_sinpos = args.ordinary_sinpos
        self.represent_length_by_lrpe = args.represent_length_by_lrpe
        self.represent_length_by_ldpe = args.represent_length_by_ldpe

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )

        ## ???
        self.embed_positions = (
            LCPositionalEmbedding(
                args.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=args.decoder_learned_pos,
            )
            if not args.no_token_positional_embeddings and self.ordinary_sinpos
            else None
        )
        # self.embed_positions_original = (
        #     PositionalEmbedding(
        #         args.max_target_positions,
        #         embed_dim,
        #         self.padding_idx,
        #         learned=args.decoder_learned_pos,
        #     )
        #     if not args.no_token_positional_embeddings and self.ordinary_sinpos
        #     else None
        # )
        # self.embed_positions_lrpe = (
        #     PositionalEmbedding(
        #         args.max_target_positions,
        #         embed_dim,
        #         self.padding_idx,
        #         learned=args.decoder_learned_pos,
        #     )
        #     if not args.no_token_positional_embeddings and self.represent_length_by_lrpe
        #     else None
        # )
        # self.embed_positions_ldpe = (
        #     PositionalEmbedding(
        #         args.max_target_positions,
        #         embed_dim,
        #         self.padding_idx,
        #         learned=args.decoder_learned_pos,
        #     )
        #     if not args.no_token_positional_embeddings and self.represent_length_by_ldpe
        #     else None
        # )

        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        if self.decoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_decoder_layer(args, no_encoder_attn)
                for _ in range(args.decoder_layers)
            ]
        )
        self.num_layers = len(self.layers)

        if args.decoder_normalize_before and not getattr(
            args, "no_decoder_final_norm", False
        ):
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        self.project_out_dim = (
            Linear(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights
            else None
        )

        self.adaptive_softmax = None
        self.output_projection = None
        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                utils.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif self.share_input_output_embed:
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = nn.Linear(
                self.output_embed_dim, len(dictionary), bias=False
            )
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.output_embed_dim ** -0.5
            )

    def build_decoder_layer(self, args, no_encoder_attn=False):
        return LCTransformerDecoderLayer(args, no_encoder_attn)

    def forward(
        self,
        prev_output_tokens,
        tgt_lengths,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            tgt_lengths,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        length,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            length,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )

    """
    A scriptable subclass of this class has an extract_features method and calls
    super().extract_features, but super() is not supported in torchscript. Aa copy of
    this function is made to be used in the subclass instead.
    """

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        length,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # embed positions
        positions = None
        if self.ordinary_sinpos:
            positions = (
                self.embed_positions(
                    prev_output_tokens, incremental_state=incremental_state
                )
                if self.embed_positions is not None
                else None
            )
            if incremental_state is not None and positions is not None:
                positions = positions[:, -1:]
        if self.represent_length_by_lrpe:
            positions_lrpe = (
                self.embed_positions(
                    prev_output_tokens,
                    incremental_state=incremental_state,
                    length=length,
                    sinpostype='ratio',
                )
                if self.embed_positions is not None
                else None
            )
            if positions_lrpe is not None:
                if incremental_state is not None:
                    positions_tmp = positions_lrpe.view(positions_lrpe.size(0), 1, -1)
                else:
                    positions_tmp = positions_lrpe
                positions = positions + positions_tmp if positions is not None else positions_tmp
        if self.represent_length_by_ldpe:
            positions_ldpe = (
                self.embed_positions(
                    prev_output_tokens,
                    incremental_state=incremental_state,
                    length=length,
                    sinpostype='absolute',
                )
                if self.embed_positions is not None
                else None
            )
            if positions_ldpe is not None:
                if incremental_state is not None:
                    positions_tmp = positions_ldpe[:, -1:]
                else:
                    positions_tmp = positions_ldpe
                positions = positions + positions_tmp if positions is not None else positions_tmp

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            # if positions is not None:
            #     positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}
