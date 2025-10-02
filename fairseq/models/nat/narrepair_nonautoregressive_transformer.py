# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from fairseq import utils

# from fairseq.iterative_refinement_generator import DecoderOut
from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat import (
    FairseqNATDecoder,
    FairseqNATEncoder,
    FairseqNATModel,
    ensemble_decoder,
    ensemble_encoder,
)
from fairseq.models.transformer import Embedding
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from typing import Any, Dict, List, Optional, Tuple
from torch import Tensor
from fairseq.modules import FairseqDropout
import pickle
from collections import namedtuple
import numpy as np
import math
import ipdb

DecoderOut = namedtuple(
    "IterativeRefinementDecoderOut",
    [
        "output_tokens",
        "output_scores",
        "attn",
        "step",
        "max_step",
        "history",
        "to_be_edited_pred",
        "wer_dur_pred",
    ],
)


def _mean_pooling(enc_feats, src_masks):
    # enc_feats: T x B x C
    # src_masks: B x T or None
    if src_masks is None:
        enc_feats = enc_feats.mean(0)
    else:
        src_masks = (~src_masks).transpose(0, 1).type_as(enc_feats)
        enc_feats = (
            (enc_feats / src_masks.sum(0)[None, :, None]) * src_masks[:, :, None]
        ).sum(0)
    return enc_feats


def _argmax(x, dim):
    return (x == x.max(dim, keepdim=True)[0]).type_as(x)


def _uniform_assignment(src_lens, trg_lens):
    max_trg_len = trg_lens.max()
    steps = (src_lens.float() - 1) / (trg_lens.float() - 1)  # step-size
    # max_trg_len
    index_t = utils.new_arange(trg_lens, max_trg_len).float()
    index_t = steps[:, None] * index_t[None, :]  # batch_size X max_trg_len
    index_t = torch.round(index_t).long().detach()
    return index_t


@register_model("nonautoregressive_transformer_cmlm")
class NATransformerModel(FairseqNATModel):
    @property
    def allow_length_beam(self):
        return True

    @staticmethod
    def add_args(parser):
        FairseqNATModel.add_args(parser)

        # length prediction
        parser.add_argument(
            "--src-embedding-copy",
            action="store_true",
            help="copy encoder word embeddings as the initial input of the decoder",
        )
        parser.add_argument(
            "--pred-length-offset",
            action="store_true",
            help="predicting the length difference between the target and source sentences",
        )
        parser.add_argument(
            "--sg-length-pred",
            action="store_true",
            help="stop the gradients back-propagated from the length predictor",
        )
        parser.add_argument(
            "--length-loss-factor",
            type=float,
            help="weights on the length prediction loss",
        )

    @classmethod
    def build_encoder(cls, args, tgt_dict, embed_tokens):
        encoder = NATransformerEncoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            encoder.apply(init_bert_params)
        return encoder

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = NATransformerDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs
    ):
        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)

        # length prediction
        length_out = self.decoder.forward_length(
            normalize=False, encoder_out=encoder_out
        )
        length_tgt = self.decoder.forward_length_prediction(
            length_out, encoder_out, tgt_tokens
        )

        # decoding
        word_ins_out = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
        )

        return {
            "word_ins": {
                "out": word_ins_out,
                "tgt": tgt_tokens,
                "mask": tgt_tokens.ne(self.pad),
                "ls": self.args.label_smoothing,
                "nll_loss": True,
            },
            "length": {
                "out": length_out,
                "tgt": length_tgt,
                "factor": self.decoder.length_loss_factor,
            },
        }

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):
        step = decoder_out.step
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history

        to_be_edited_pred = decoder_out.to_be_edited_pred
        wer_dur_pred = decoder_out.wer_dur_pred

        for_wer_gather = wer_dur_pred.cumsum(dim=-1)
        for_wer_gather = (
            torch.nn.functional.one_hot(
                for_wer_gather, num_classes=for_wer_gather.max() + 1
            )[:, :-1, :-1]
            .sum(-2)
            .cumsum(dim=-1)
        )
        # execute the decoder
        output_masks = output_tokens.ne(self.pad)
        decoderout, pre_feature, _ = self.decoder(
            normalize=True,
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out,
            step=step,
            # wer_dur=wer_dur_pred,
            # to_be_edited=to_be_edited_pred,
            for_wer_gather=for_wer_gather,
        )
        _scores, _tokens = torch.softmax(decoderout, -1).max(-1)
        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])
        if history is not None:
            history.append(output_tokens.clone())

        return (
            decoder_out._replace(
                output_tokens=output_tokens,
                output_scores=output_scores,
                attn=None,
                history=history,
            ),
            pre_feature,
        )

    def initialize_output_tokens(
        self,
        encoder_out,
        src_tokens,
        edit_thre=0.0,
        print_werdur=False,
        werdur_gt_str="",
    ):
        # length prediction
        # length_tgt = self.decoder.forward_length_prediction(
        #     self.decoder.forward_length(normalize=True, encoder_out=encoder_out),
        #     encoder_out=encoder_out,
        # )
        # if len(length_tgt.size()) == 0:
        #     length_tgt = length_tgt.unsqueeze(0)

        (
            wer_dur_pred,
            to_be_edited_pred,
            closest_pred,
        ) = self.decoder.forward_wer_dur_and_tbe(
            normalize=False, encoder_out=encoder_out
        )
        if "log" in self.werdur_loss_type:
            wer_dur_pred = (
                (torch.exp(wer_dur_pred) - 1.0).squeeze(-1).round().long().clamp_(min=0)
            )
            length_tgt = wer_dur_pred.sum(-1)
        else:
            wer_dur_pred = wer_dur_pred.squeeze(-1).round().long().clamp_(min=0)
            length_tgt = wer_dur_pred.sum(-1)

        max_length = length_tgt.clamp_(min=2).max()
        idx_length = utils.new_arange(src_tokens, max_length)

        initial_output_tokens = src_tokens.new_zeros(
            src_tokens.size(0), max_length
        ).fill_(self.pad)
        initial_output_tokens.masked_fill_(
            idx_length[None, :] < length_tgt[:, None], self.unk
        )
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out["encoder_out"][0])

        return (
            DecoderOut(
                output_tokens=initial_output_tokens,
                output_scores=initial_output_scores,
                attn=None,
                step=0,
                max_step=0,
                history=None,
                to_be_edited_pred=to_be_edited_pred,
                wer_dur_pred=wer_dur_pred,
            ),
            encoder_out,
        )

    def regenerate_length_beam(self, decoder_out, beam_size):
        output_tokens = decoder_out.output_tokens
        length_tgt = output_tokens.ne(self.pad).sum(1)
        length_tgt = (
            length_tgt[:, None]
            + utils.new_arange(length_tgt, 1, beam_size)
            - beam_size // 2
        )
        length_tgt = length_tgt.view(-1).clamp_(min=2)
        max_length = length_tgt.max()
        idx_length = utils.new_arange(length_tgt, max_length)

        initial_output_tokens = output_tokens.new_zeros(
            length_tgt.size(0), max_length
        ).fill_(self.pad)
        initial_output_tokens.masked_fill_(
            idx_length[None, :] < length_tgt[:, None], self.unk
        )
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(decoder_out.output_scores)

        return decoder_out._replace(
            output_tokens=initial_output_tokens, output_scores=initial_output_scores
        )


class NATransformerDecoder(FairseqNATDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(
            args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn
        )
        self.dictionary = dictionary
        self.bos = dictionary.bos()
        self.unk = dictionary.unk()
        self.eos = dictionary.eos()

        self.chunk_size = args.decoder_embed_dim
        # self.link_positional = PositionalEmbedding(
        #     args.max_target_positions, args.decoder_embed_dim, self.padding_idx, True
        # )
        self.query_linear = nn.Linear(args.decoder_embed_dim, args.decoder_embed_dim)
        self.key_linear = nn.Linear(args.decoder_embed_dim, args.decoder_embed_dim)
        self.value_linear = nn.Linear(args.decoder_embed_dim, args.decoder_embed_dim)

        ft_node_types = pickle.load(
            open(
                "/mnt/hdd/yzy/yangzhenyu/RenewNAT_code/defect4j-bin/all_node_types.pkl",
                "rb",
            )
        )
        self.node_num = len(ft_node_types)
        self.treefeature = Biaffine(
            args.decoder_embed_dim, args.decoder_embed_dim, self.node_num
        )
        # self.treelinear = nn.Linear(args.decoder_embed_dim, self.node_num)
        # self.interattention = InterAttentionLayer(args.decoder_embed_dim, 1, args.dropout)
        self.sg_length_pred = getattr(args, "sg_length_pred", False)
        self.pred_length_offset = getattr(args, "pred_length_offset", False)
        self.length_loss_factor = getattr(args, "length_loss_factor", 1)
        self.src_embedding_copy = getattr(args, "src_embedding_copy", False)
        # if self.length_loss_factor > 0:
        #     self.embed_length = Embedding(
        #         getattr(args, "max_target_positions", 256), self.encoder_embed_dim, None
        #     )
        #     torch.nn.init.normal_(self.embed_length.weight, mean=0, std=0.02)
        if self.src_embedding_copy:
            self.copy_attn = torch.nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        self.mlm_layers = getattr(args, "mlm_layers", 4)
        # for p in self.parameters():
        #     p.requires_grad = False
        self.encoder_embed_dim = args.encoder_embed_dim
        self.dur_predictor = DurationPredictor(
            idim=self.encoder_embed_dim,
            n_layers=5,
            n_chans=self.encoder_embed_dim,
            ffn_layers=2,
            ln_eps=1e-5,
            remove_edit_emb=False,
            to_be_edited_size=1,
        )

    @ensemble_decoder
    def forward(
        self,
        normalize,
        encoder_out,
        prev_output_tokens,
        step=0,
        for_wer_gather=None,
        pre_feature=None,
        **unused
    ):
        # ipdb.set_trace()
        features, tree_link, _ = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            embedding_copy=(step == 0) & self.src_embedding_copy,
            for_wer_gather=for_wer_gather,
            pre_feature=pre_feature,
        )

        # tree_link, tree_feature = self.ast_link(
        #     features,
        #     # prev_output_tokens,
        #     # self.link_positional,
        #     self.query_linear,
        #     self.key_linear,
        #     self.value_linear,
        #     # self.treelinear,
        # )
        decoder_out = self.output_layer(features)
        return (
            F.log_softmax(decoder_out, -1) if normalize else decoder_out,
            features,
            tree_link,
        )
        # decoder_out = self.output_layer(features)
        # return F.log_softmax(decoder_out, -1) if normalize else decoder_out

    # @ensemble_decoder
    # def forward_length(self, normalize, encoder_out):
    #     # enc_feats = encoder_out["length_out"][0].squeeze(0)  # T x B x C
    #     enc_feats = encoder_out["encoder_out"][0]
    #     if len(encoder_out["encoder_padding_mask"]) > 0:
    #         src_masks = encoder_out["encoder_padding_mask"][0]  # B x T
    #     else:
    #         src_masks = None
    #     enc_feats = _mean_pooling(enc_feats, src_masks)
    #     if self.sg_length_pred:
    #         enc_feats = enc_feats.detach()
    #     length_out = F.linear(enc_feats, self.embed_length.weight)
    #     # length_out[:, 0] += float('-inf')
    #     return F.log_softmax(length_out, -1) if normalize else length_out

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out=None,
        early_exit=None,
        embedding_copy=False,
        for_wer_gather=None,
        pre_feature=None,
        **unused
    ):
        """
        Similar to *forward* but only return features.

        Inputs:
            prev_output_tokens: Tensor(B, T)
            encoder_out: a dictionary of hidden states and masks

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
            the LevenshteinTransformer decoder has full-attention to all generated tokens
        """
        positions = (
            self.embed_positions(prev_output_tokens)
            if self.embed_positions is not None
            else None
        )
        tree_link = None
        # embedding
        # ipdb.set_trace()
        if embedding_copy:
            src_embd = encoder_out["encoder_embedding"][0]
            if len(encoder_out["encoder_padding_mask"]) > 0:
                src_mask = encoder_out["encoder_padding_mask"][0]
            else:
                src_mask = None

            bsz, seq_len = prev_output_tokens.size()
            attn_score = torch.bmm(
                self.copy_attn(positions),
                (src_embd + encoder_out["encoder_pos"][0]).transpose(1, 2),
            )
            if src_mask is not None:
                attn_score = attn_score.masked_fill(
                    src_mask.unsqueeze(1).expand(-1, seq_len, -1), float("-inf")
                )
            attn_weight = F.softmax(attn_score, dim=-1)
            x = torch.bmm(attn_weight, src_embd)
            wer_dur_embedding = self.forward_wer_dur_embedding(
                src_embd,
                src_mask,
                prev_output_tokens.ne(self.padding_idx),
                for_wer_gather,
            )
            mask_target_x, decoder_padding_mask = self.forward_embedding(
                prev_output_tokens,
                wer_dur_embedding,
            )
            output_mask = prev_output_tokens.eq(self.unk)
            cat_x = torch.cat([mask_target_x.unsqueeze(2), x.unsqueeze(2)], dim=2).view(
                -1, x.size(2)
            )
            x = cat_x.index_select(
                dim=0,
                index=torch.arange(bsz * seq_len).cuda() * 2
                + output_mask.view(-1).long(),
            ).reshape(bsz, seq_len, x.size(2))

        else:
            x, decoder_padding_mask = self.forward_embedding(prev_output_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        positions = positions.transpose(0, 1)
        attn = None
        inner_states = [x]

        # Control NAT decoder module and MLM module layers
        # full_mask_layers = [0, 1, 2, 3]
        full_mask_layers = [i for i in range(len(self.layers) - self.mlm_layers)]
        # decoder layers
        for i, layer in enumerate(self.layers):
            if embedding_copy and i not in full_mask_layers:
                break
            if not embedding_copy and i in full_mask_layers:
                continue
            # early exit from the decoder.
            if (early_exit is not None) and (i >= early_exit):
                break

            # if positions is not None and (i==0 or embedding_copy):
            if positions is not None and (
                i == 0 or i == len(full_mask_layers) or embedding_copy
            ):
                x += positions
                x = self.dropout_module(x)

            x, attn, _ = layer(
                x,
                encoder_out["encoder_out"][0]
                if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                else None,
                encoder_out["encoder_padding_mask"][0]
                if (
                    encoder_out is not None
                    and len(encoder_out["encoder_padding_mask"]) > 0
                )
                else None,
                self_attn_mask=None,
                self_attn_padding_mask=decoder_padding_mask,
            )
            inner_states.append(x)
        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        if embedding_copy:
            tree_link, x = self.ast_link(
                x,
                self.query_linear,
                self.key_linear,
                self.value_linear,
            )
        if self.project_out_dim is not None:
            x = self.project_out_dim(x)
        cos=[]
        for i in range(len(x)):
            for j in range(len(tree_link)):
                cos_sim = F.cosine_similarity(x[i].unsqueeze(0), tree_link[j].unsqueeze(0))
                cos.append(cos_sim)
        print(cos)
        return x, tree_link, {"attn": attn, "inner_states": inner_states}

    def forward_embedding(self, prev_output_tokens, states=None):
        positions = (
            self.embed_positions(prev_output_tokens)
            if self.embed_positions is not None
            else None
        )
        # embed tokens and positions
        if states is None:
            x = self.embed_tokens(prev_output_tokens)
            if self.project_in_dim is not None:
                x = self.project_in_dim(x)
        else:
            x = states

        # if positions is not None:
        #     x += positions
        # x = self.dropout_module(x)
        decoder_padding_mask = prev_output_tokens.eq(self.padding_idx)
        return x, decoder_padding_mask

    def forward_wer_dur_embedding(
        self,
        src_embeds,
        src_masks,
        tgt_masks,
        for_wer_gather=None,
    ):
        # ipdb.set_trace()
        batch_size, _, hidden_size = src_embeds.shape

        for_wer_gather = for_wer_gather[:, :, None].long()

        to_reshape = torch.gather(
            src_embeds, 1, for_wer_gather.repeat(1, 1, src_embeds.shape[2])
        )

        to_reshape = to_reshape * tgt_masks[:, :, None]

        return to_reshape

    @ensemble_decoder
    def forward_wer_dur_and_tbe(self, normalize, encoder_out):
        # ipdb.set_trace()
        assert len(encoder_out["encoder_out"]) == 1
        enc_feats = encoder_out["encoder_out"][0]  # T x B x C
        src_masks = encoder_out["encoder_padding_mask"][0]  # B x T or None
        encoder_embedding = encoder_out["encoder_embedding"][0]  # B x T x C
        enc_feats = enc_feats.transpose(0, 1)
        # enc_feats = _mean_pooling(enc_feats, src_masks)
        if self.sg_length_pred:
            enc_feats = enc_feats.detach()
        src_masks = ~src_masks
        wer_dur_out, to_be_edited_out = self.dur_predictor(enc_feats, src_masks)
        closest = None

        return wer_dur_out, to_be_edited_out, closest

    def forward_copying_source(self, src_embeds, src_masks, tgt_masks):
        length_sources = src_masks.sum(1)
        length_targets = tgt_masks.sum(1)
        mapped_inputs = _uniform_assignment(length_sources, length_targets).masked_fill(
            ~tgt_masks, 0
        )
        copied_embedding = torch.gather(
            src_embeds,
            1,
            mapped_inputs.unsqueeze(-1).expand(
                *mapped_inputs.size(), src_embeds.size(-1)
            ),
        )
        return copied_embedding

    def forward_length_prediction(self, length_out, encoder_out, tgt_tokens=None):
        enc_feats = encoder_out["encoder_out"][0]  # T x B x C
        if len(encoder_out["encoder_padding_mask"]) > 0:
            src_masks = encoder_out["encoder_padding_mask"][0]  # B x T
        else:
            src_masks = None
        if self.pred_length_offset:
            if src_masks is None:
                src_lengs = enc_feats.new_ones(enc_feats.size(1)).fill_(
                    enc_feats.size(0)
                )
            else:
                src_lengs = (~src_masks).transpose(0, 1).type_as(enc_feats).sum(0)
            src_lengs = src_lengs.long()

        if tgt_tokens is not None:
            # obtain the length target
            tgt_lengs = tgt_tokens.ne(self.padding_idx).sum(1).long()
            if self.pred_length_offset:
                length_tgt = tgt_lengs - src_lengs + 128
            else:
                length_tgt = tgt_lengs
            length_tgt = length_tgt.clamp(min=0, max=1023)

        else:
            # predict the length target (greedy for now)
            # TODO: implementing length-beam
            pred_lengs = length_out.max(-1)[1]
            if self.pred_length_offset:
                length_tgt = pred_lengs - 128 + src_lengs
            else:
                length_tgt = pred_lengs

        return length_tgt

    def ast_link(
        self,
        features,
        # prev_output_tokens,
        # link_positional,
        query_linear,
        key_linear,
        value_linear,
        # treelinear,
    ):
        # ipdb.set_trace()
        # links_feature = link_positional(prev_output_tokens)
        # features_withpos = torch.cat([features, links_feature], dim=-1)
        batch_size = features.shape[0]
        prelen = features.shape[1]
        query_chunks = query_linear(features).reshape(
            batch_size, prelen, self.chunk_size
        )
        key_chunks = key_linear(features).reshape(batch_size, prelen, self.chunk_size)
        value_chunks = value_linear(features).reshape(
            batch_size, prelen, self.chunk_size
        )
        tree_link = self.treefeature(query_chunks, key_chunks)
        # tree_link = treelinear(treefeature)
        # attention_scores = torch.matmul(query_chunks, key_chunks.permute(0, 2,1))/ math.sqrt(self.args.decoder_embed_dim)
        # attention_scores = F.softmax(attention_scores, dim = -1)
        # hiddenout = torch.matmul(attention_scores, value_chunks)
        # treeout = self.interattention(features,features,features,treefeature)
        # treefeature = torch.sum(treefeature, dim=2).squeeze(2)
        # treeout = features + treefeature
        attention_scores = torch.matmul(query_chunks, key_chunks.permute(0, 2, 1))
        attention_scores = attention_scores / math.sqrt(self.chunk_size)

        attention_probs = F.softmax(attention_scores, dim=-1)

        context = torch.matmul(attention_probs, value_chunks)
        return tree_link, context


class NATransformerEncoder(FairseqNATEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        # self.len_embed = torch.nn.Parameter(torch.Tensor(1, args.encoder_embed_dim))
        # torch.nn.init.normal_(self.len_embed, mean=0, std=0.02)

    @ensemble_encoder
    def forward(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()

        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        encoder_pos = self.embed_positions(src_tokens)
        # account for padding while computing the representation
        if encoder_padding_mask is not None:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # x = torch.cat([self.len_embed.unsqueeze(0).repeat(src_tokens.size(0),1,1), x], dim=1)
        x = self.dropout_module(x)
        # if encoder_padding_mask is not None:
        #     encoder_padding_mask = torch.cat(
        #         [encoder_padding_mask.new(src_tokens.size(0), 1).fill_(0), encoder_padding_mask], dim=1)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []

        if return_all_hiddens:
            encoder_states.append(x)
        # encoder layers
        for layer in self.layers:
            x = layer(
                x, encoder_padding_mask=encoder_padding_mask if has_pads else None
            )
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)
        len_x = x[0, :, :]
        # x = x[1:, :, :]
        # if encoder_padding_mask is not None:
        #     encoder_padding_mask = encoder_padding_mask[:, 1:]
        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_pos": [encoder_pos],
            "length_out": [len_x],
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

    def forward_embedding(
        self, src_tokens, token_embedding: Optional[torch.Tensor] = None
    ):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        x = embed = token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        # x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["length_out"]) == 0:
            new_length_out = []
        else:
            new_length_out = [encoder_out["length_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [
                encoder_out["encoder_embedding"][0].index_select(0, new_order)
            ]
        if len(encoder_out["encoder_pos"]) == 0:
            new_encoder_pos = []
        else:
            new_encoder_pos = [encoder_out["encoder_pos"][0].index_select(0, new_order)]

        if len(encoder_out["src_tokens"]) == 0:
            src_tokens = []
        else:
            src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]

        if len(encoder_out["src_lengths"]) == 0:
            src_lengths = []
        else:
            src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_pos": new_encoder_pos,
            "length_out": new_length_out,
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": src_tokens,  # B x T
            "src_lengths": src_lengths,  # B x 1
        }


class Biaffine(nn.Module):
    def __init__(self, in1_features, in2_features, out_features, bias=(True, True)):
        super(Biaffine, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.bias = bias
        self.linear_input_size = in1_features + int(bias[0])
        self.linear_output_size = out_features * (in2_features + int(bias[1]))
        self.linear = nn.Linear(
            in_features=self.linear_input_size,
            out_features=self.linear_output_size,
            bias=False,
        )
        self.reset_parameters()

    def reset_parameters(self):
        W = np.zeros(
            (self.linear_output_size, self.linear_input_size), dtype=np.float32
        )
        self.linear.weight.data.copy_(torch.from_numpy(W))

    def forward(self, input1, input2):
        # #ipdb.set_trace()
        batch_size, len1, dim1 = input1.size()
        batch_size, len2, dim2 = input2.size()
        if self.bias[0]:
            ones = input1.data.new(batch_size, len1, 1).zero_().fill_(1)
            input1 = torch.cat((input1, Variable(ones)), dim=2)
            dim1 += 1
        if self.bias[1]:
            ones = input2.data.new(batch_size, len2, 1).zero_().fill_(1)
            input2 = torch.cat((input2, Variable(ones)), dim=2)
            dim2 += 1

        affine = self.linear(input1)

        affine = affine.view(batch_size, len1 * self.out_features, dim2)
        input2 = torch.transpose(input2, 1, 2)

        biaffine = torch.transpose(torch.bmm(affine, input2), 1, 2)

        biaffine = biaffine.contiguous().view(batch_size, len2, len1, self.out_features)

        return biaffine


class LayerNorm(torch.nn.LayerNorm):
    """Layer normalization module.
    :param int nout: output dim size
    :param int dim: dimension to be normalized
    """

    def __init__(self, nout, dim=-1, eps=1e-12):
        """Construct an LayerNorm object."""
        super(LayerNorm, self).__init__(nout, eps=eps)
        self.dim = dim

    def forward(self, x):
        """Apply layer normalization.
        :param torch.Tensor x: input tensor
        :return: layer normalized tensor
        :rtype torch.Tensor
        """
        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        return super(LayerNorm, self).forward(x.transpose(1, -1)).transpose(1, -1)


class DurationPredictor(torch.nn.Module):
    def __init__(
        self,
        idim,
        n_layers=2,
        n_chans=384,
        kernel_size=3,
        dropout_rate=0.1,
        ffn_layers=1,
        offset=1.0,
        ln_eps=1e-12,
        remove_edit_emb=False,
        to_be_edited_size=1,
        padding="SAME",
    ):
        """Initilize duration predictor module.
        Args:
            idim (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
            offset (float, optional): Offset value to avoid nan in log domain.
        """
        super(DurationPredictor, self).__init__()
        #'''
        self.offset = offset
        self.conv = torch.nn.ModuleList()
        self.kernel_size = kernel_size
        self.padding = padding
        self.remove_edit_emb = remove_edit_emb
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        in_chans, n_chans, kernel_size, stride=1, padding=0
                    ),
                    torch.nn.ReLU(),
                    LayerNorm(n_chans, dim=1, eps=ln_eps),
                    FairseqDropout(dropout_rate, module_name="DP_dropout"),
                )
            ]
        if ffn_layers == 1:
            self.werdur_linear = torch.nn.Linear(n_chans, 1)
            self.edit_linear = torch.nn.Linear(n_chans, to_be_edited_size)
        else:
            assert ffn_layers == 2
            self.werdur_linear = torch.nn.Sequential(
                torch.nn.Linear(n_chans, n_chans // 2),
                torch.nn.ReLU(),
                FairseqDropout(dropout_rate, module_name="DP_dropout"),
                torch.nn.Linear(n_chans // 2, 1),
            )
            self.edit_linear = torch.nn.Sequential(
                torch.nn.Linear(n_chans, n_chans // 2),
                torch.nn.ReLU(),
                FairseqDropout(dropout_rate, module_name="DP_dropout"),
                torch.nn.Linear(n_chans // 2, to_be_edited_size),
                torch.nn.Sigmoid(),
            )
        #'''
        # self.werdur_linear = torch.nn.Linear(idim, 1)
        # self.edit_linear = torch.nn.Linear(idim, 1)

    def forward(self, xs, x_nonpadding=None):
        #'''
        # ipdb.set_trace()
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            if self.padding == "SAME":
                xs = F.pad(xs, [self.kernel_size // 2, self.kernel_size // 2])
            elif self.padding == "LEFT":
                xs = F.pad(xs, [self.kernel_size - 1, 0])
            xs = f(xs)  # (B, C, Tmax)
            if x_nonpadding is not None:
                xs = xs * x_nonpadding[:, None, :]

        xs = xs.transpose(1, -1)
        #'''
        werdur = self.werdur_linear(xs) * x_nonpadding[:, :, None]  # (B, Tmax)
        to_be_edited = self.edit_linear(xs) * x_nonpadding[:, :, None]  # (B, Tmax

        return werdur, to_be_edited


@register_model_architecture(
    "nonautoregressive_transformer_cmlm", "nonautoregressive_transformer_cmlm"
)
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.layer_norm = getattr(args, "layer_norm", "normal")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # --- special arguments ---
    args.sg_length_pred = getattr(args, "sg_length_pred", False)
    args.pred_length_offset = getattr(args, "pred_length_offset", False)
    args.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
    args.src_embedding_copy = getattr(args, "src_embedding_copy", False)


@register_model_architecture(
    "nonautoregressive_transformer_cmlm", "nonautoregressive_transformer_cmlm_wmt_en_de"
)
def nonautoregressive_transformer_wmt_en_de(args):
    base_architecture(args)
