# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from fairseq import utils
from fairseq.iterative_refinement_generator import DecoderOut
from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat import FairseqNATModel
from fairseq.modules.transformer_sentence_encoder import init_bert_params
import torch.nn.functional as F

from fairseq.models.nat.narrepair_nonautoregressive_transformer import (
    NATransformerEncoder,
    NATransformerDecoder,
    NATransformerModel,
)
import logging
import random
import copy
from contextlib import contextmanager
from fairseq.utils import new_arange
from fairseq.modules import FairseqDropout
import math
import ipdb

logger = logging.getLogger(__name__)


def _skeptical_unmasking(output_scores, output_masks, p):
    sorted_index = output_scores.sort(-1)[1]
    boundary_len = (
        (output_masks.sum(1, keepdim=True).type_as(output_scores) - 2) * p
    ).long()
    skeptical_mask = new_arange(output_masks) < boundary_len
    return skeptical_mask.scatter(1, sorted_index, skeptical_mask)


@contextmanager
def torch_seed(seed):
    state = torch.random.get_rng_state()
    state_cuda = torch.cuda.random.get_rng_state()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(state)
        torch.cuda.random.set_rng_state(state_cuda)


@register_model("narrepair")
class NARrepair(FairseqNATModel):
    # forward_decoder = NATransformerModel.forward_decoder
    nat_forward_decoder = NATransformerModel.forward_decoder
    initialize_output_tokens = NATransformerModel.initialize_output_tokens
    regenerate_length_beam = NATransformerModel.regenerate_length_beam

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.werdur_max_predict = getattr(args, "werdur_max_predict", 5.0)
        print("werdur_max_predict: ", self.werdur_max_predict)
        self.werdur_loss_type = getattr(args, "werdur_loss_type", "l2")
        if self.werdur_loss_type == "l2":
            self.werdur_loss_func = F.mse_loss
        elif self.werdur_loss_type == "log_l2":
            self.werdur_loss_func = self.log_mse_loss
        elif self.werdur_loss_type == "l1":
            self.werdur_loss_func = F.l1_loss
        elif self.werdur_loss_type == "log_l1":
            self.werdur_loss_func = self.log_l1_loss

    def log_mse_loss(self, hypo, ref, reduction="none"):
        hypo = torch.exp(hypo) - 1.0
        return F.mse_loss(hypo, ref, reduction=reduction)

    def log_l1_loss(self, hypo, ref, reduction="none"):
        hypo = torch.exp(hypo) - 1.0
        return F.l1_loss(hypo, ref, reduction=reduction)

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
        parser.add_argument(
            "--werdur-loss-type",
            type=str,
            help="type of werdur loss",
        )
        parser.add_argument(
            "--werdur-max-predict",
            type=float,
            help="max value of werdur",
        )

    @classmethod
    def build_encoder(cls, args, tgt_dict, embed_tokens):
        # #ipdb.set_trace()
        encoder = NATransformerEncoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            encoder.apply(init_bert_params)
        return encoder

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        # #ipdb.set_trace()
        decoder = NATransformerDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        tgt_tokens,
        glat=None,
        wer_dur=None,
        to_be_edited=None,
        for_wer_gather=None,
        treepath=None,
        **kwargs
    ):
        # ipdb.set_trace()
        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)

        # length prediction
        # length_out = self.decoder.forward_length(
        #     normalize=False, encoder_out=encoder_out
        # )
        # length_tgt = self.decoder.forward_length_prediction(  # 预测目标语言每个句子的长度
        #     length_out, encoder_out, tgt_tokens
        # )
        (
            wer_dur_pred,
            to_be_edited_pred,
            closest_pred,
        ) = self.decoder.forward_wer_dur_and_tbe(
            normalize=False, encoder_out=encoder_out
        )
        elements = [str(x) for x in wer_dur_pred.tolist()]
        wer_dur_str = ' '.join(elements)
        elements = [str(x) for x in wer_dur_pred.tolist()]
        edited_str = ' '.join(elements)
        file = open('predicted_data.txt', 'w')
        add_str = wer_dur_str+"|||"+edited_str
        file.write(add_str+'\n')
        # wer_dur表示预测的长度，to_be_edited表示长度预测中符号的正负
        wer_dur = wer_dur.type_as(wer_dur_pred).clamp(
            0.0, self.werdur_max_predict
        )  # modify wer_dur is ok because in decoder only use for gather
        # 讲wer_due的类型转化为和wer_dur_pred一样的类型
        src_no_pad = ~(encoder_out["encoder_padding_mask"][0])  # 为单词数量+2的true

        wer_dur_pred = wer_dur_pred.squeeze(-1)

        wer_dur_pred_loss_float = self.werdur_loss_func(
            wer_dur_pred, wer_dur, reduction="none"
        ).float()
        wer_dur_pred_loss = (
            wer_dur_pred_loss_float[src_no_pad.bool()].mean().type_as(wer_dur_pred)
        )

        to_be_edited_pred_loss_float = F.binary_cross_entropy_with_logits(
            to_be_edited_pred.squeeze(-1),
            to_be_edited.type_as(to_be_edited_pred),
            reduction="none",
        ).float()
        to_be_edited_pred_loss = (
            to_be_edited_pred_loss_float[src_no_pad.bool()]
            .mean()
            .type_as(to_be_edited_pred)
        )

        nonpad_positions = tgt_tokens.ne(self.pad)  # 标记目标句子的mask
        seq_lens = (nonpad_positions).sum(1)  # batch中每个句子的长度
        rand_seed = random.randint(0, 19260817)  # 定义随机种子
        # import pdb
        # ipdb.set_trace()
        # glancing sampling
        glat_info = None
        ori_tgt_tokens = tgt_tokens
        if glat and tgt_tokens is not None:
            with torch.no_grad():
                with torch_seed(rand_seed):
                    (
                        word_ins_out_first,
                        _,
                        _,
                    ) = self.decoder(  # 维度为batchsize*sequencelen*字典长度
                        normalize=False,
                        prev_output_tokens=prev_output_tokens,
                        encoder_out=encoder_out,
                        for_wer_gather=for_wer_gather,
                    )
                pred_tokens = word_ins_out_first.argmax(-1)  # 将概率值最大的坐标提取出来
                same_num = ((pred_tokens == tgt_tokens) & nonpad_positions).sum(1)
                input_mask = torch.ones_like(nonpad_positions)
                bsz, seq_len = tgt_tokens.size()
                for li in range(bsz):
                    target_num = (
                        ((seq_lens[li] - same_num[li].sum()).float())
                        * glat["context_p"]
                    ).long()  # glat表示掩盖一定比例的单词
                    if target_num > 0:
                        input_mask[li].scatter_(
                            dim=0,
                            index=torch.randperm(seq_lens[li])[:target_num].cuda(),
                            value=0,
                        )  # 生成随机数，随机数表示掩盖单词的位置
                input_mask = input_mask.eq(1)
                input_mask = input_mask.masked_fill(~nonpad_positions, False)
                glat_prev_output_tokens = prev_output_tokens.masked_fill(
                    ~input_mask, 0
                ) + tgt_tokens.masked_fill(
                    input_mask, 0
                )  # 第一次带掩码的目标数据
                glat_tgt_tokens = tgt_tokens.masked_fill(~input_mask, self.pad)

                prev_output_tokens, tgt_tokens = (
                    glat_prev_output_tokens,
                    glat_tgt_tokens,
                )
                # #ipdb.set_trace()
                glat_info = {
                    "glat_accu": (same_num.sum() / seq_lens.sum()).item(),  # 预测准确率
                    "glat_context_p": glat["context_p"],
                }

        with torch_seed(rand_seed):
            word_ins_out, pre_feature, tree_link = self.decoder(
                normalize=False,
                prev_output_tokens=prev_output_tokens,
                encoder_out=encoder_out,
                for_wer_gather=for_wer_gather,
            )
        # ipdb.set_trace()
        # Get MLM module input by random masking the tgt_tokens
        target_masks = (
            ori_tgt_tokens.ne(self.pad)
            & ori_tgt_tokens.ne(self.bos)
            & ori_tgt_tokens.ne(self.eos)
        )
        target_score = (
            ori_tgt_tokens.clone().float().uniform_()
        )  # clone()为复制一个向量，uniform_()从均匀分布中抽样数值就行填充
        target_score.masked_fill_(~target_masks, math.inf)  # 将开头与结尾进行遮盖
        target_length = target_masks.sum(1).float()  # 获取目标句子长度
        target_length = target_length * target_length.clone().uniform_()
        target_length = target_length + 1  # make sure to mask at least one token.
        _, target_rank = target_score.sort(1)  # 按照target_score的分数进行排序
        target_cutoff = new_arange(target_rank) < target_length[:, None].long()
        output_tokens = ori_tgt_tokens.masked_fill(
            target_cutoff.scatter(1, target_rank, target_cutoff), self.unk
        )
        new_feature = pre_feature.clone()
        # Use MLM module to predict
        with torch_seed(rand_seed):
            word_ins_out_cmlm, _, _ = self.decoder(
                normalize=False,
                prev_output_tokens=output_tokens,
                encoder_out=encoder_out,
                # The mark for MLM module
                for_wer_gather=for_wer_gather,
                step=1,
                pre_feature=new_feature,
            )
            word_ins_mask = output_tokens.eq(self.unk)
        tree_link = tree_link[:, 1:-1, 1:-1]
        tree_link = tree_link.reshape(-1, tree_link.size(-1))
        # tree_link_cmlm = tree_link_cmlm[:, 1:-1, 1:-1]
        # tree_link_cmlm = tree_link_cmlm.reshape(-1, tree_link_cmlm.size(-1))
        treepath = treepath.reshape(-1)
        # ipdb.set_trace()
        # lenth = int(treepath.shape[1] ** 0.5)
        # treepath = treepath.reshape(treepath.shape[0], lenth, lenth)
        # ipdb.set_trace()
        ret = {
            # NAT decoder loss
            "word_ins1": {
                "out": word_ins_out,
                "tgt": tgt_tokens,
                "mask": tgt_tokens.ne(self.pad),
                "ls": self.args.label_smoothing,
                "nll_loss": True,
            },
            # "tree_ins1": {
            #     "out": tree_link,
            #     "tgt": treepath.long(),
            #     "mask": tgt_tokens.ne(self.pad),
            #     "ls": self.args.label_smoothing,
            #     "factor": 0.1,
            # },
            # MLM module loss
            "word_ins2": {
                "out": word_ins_out_cmlm,
                "tgt": ori_tgt_tokens,
                "mask": word_ins_mask,
                "ls": self.args.label_smoothing,
                "nll_loss": True,
            },
            "tree_ins2": {
                "out": tree_link,
                "tgt": treepath.long(),
                # "mask": word_ins_mask,
                # "ls": self.args.label_smoothing,
                "factor": 0.1,
            },
            # Length loss
            "length1": {
                "loss": wer_dur_pred_loss,
                "factor": 0.5,
            },
            "length2": {
                "loss": to_be_edited_pred_loss,
                "factor": 0.5,
            },
        }
        # #ipdb.set_trace()
        if glat_info is not None:
            ret.update(glat_info)
        return ret

    def cmlm_forward_decoder(
        self, decoder_out, encoder_out, decoding_format=None, **kwargs
    ):
        # ##ipdb.set_trace()
        print("----------cmlm_forward_decoder--------------")
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history

        # execute the decoder
        output_masks = output_tokens.eq(self.unk)
        _scores, _tokens = torch.softmax(
            self.decoder(
                normalize=False,
                prev_output_tokens=output_tokens,
                encoder_out=encoder_out,
                step=1,
            ),
            -1,
        ).max(-1)
        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])

        if history is not None:
            history.append(output_tokens.clone())

        return decoder_out._replace(
            output_tokens=output_tokens,
            # output_tokens=_tokens,
            output_scores=output_scores,
            # output_scores=_scores,
            attn=None,
            history=history,
        )

    def forward_encoder(self, encoder_inputs):
        src_tokens, src_lengths = encoder_inputs

        return self.encoder(src_tokens, src_lengths=src_lengths)

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):
        # Generate the potential results from NAT decoder module
        # ipdb.set_trace()
        decoder_out_first, pre_feature = self.nat_forward_decoder(
            decoder_out, encoder_out, decoding_format=None, **kwargs
        )
        output_tokens = decoder_out_first.output_tokens
        output_scores = decoder_out_first.output_scores
        history = decoder_out.history
        # mask the tokens whose confidence is below α
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
        skeptical_mask = output_scores.lt(0.7) & output_tokens.ne(self.pad)
        output_tokens.masked_fill_(skeptical_mask, self.unk)
        output_scores.masked_fill_(skeptical_mask, 0.0)
        # execute MLM module
        output_masks = output_tokens.eq(self.unk)
        new_feature = pre_feature.clone()
        decoderout, _, _ = self.decoder(
            normalize=False,
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out,
            step=1,
            for_wer_gather=for_wer_gather,
            pre_feature=new_feature,
        )

        _scores, _tokens = torch.softmax(decoderout, -1).max(-1)
        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])
        if history is not None:
            history.append(output_tokens.clone())
        # return the result
        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history,
        )


@register_model_architecture("narrepair", "narrepair_6e6d512")
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
    # cmlm dropout参数
    args.cmlm_dropout = getattr(args, "cmlm_dropout", 0.2)


@register_model_architecture("narrepair", "narrepair")
def glat_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(
        args, "encoder_ffn_embed_dim", args.encoder_embed_dim * 2
    )
    args.encoder_attention_heads = getattr(
        args, "encoder_attention_heads", args.encoder_embed_dim // 64
    )

    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.decoder_embed_dim * 2
    )
    args.decoder_attention_heads = getattr(
        args, "decoder_attention_heads", args.decoder_embed_dim // 64
    )
    base_architecture(args)


@register_model_architecture("narrepair", "narrepair_base")
def base_architecture2(args):
    base_architecture(args)
