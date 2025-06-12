from math import sqrt
import math
import torch
import torch.nn as nn
from transformers import CLIPModel
import pdb
import numpy as np
from utils.functions import sinkhorn_torch, cosine_sim
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.args = args
        self.clip = CLIPModel.from_pretrained(self.args.pretrained_model)

        self.text_tokens_fc = nn.Linear(
            self.args.model.input_hidden_dim, self.args.model.input_hidden_dim
        )
        self.image_tokens_fc = nn.Linear(
            self.args.model.input_image_hidden_dim, self.args.model.input_hidden_dim
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        pixel_values=None,
    ):
        clip_output = self.clip(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )

        text_seq_tokens = clip_output.text_model_output[0]  # [batch_size, 40, 512]
        image_patch_tokens = clip_output.vision_model_output[0]  # [batch_size, 50, 768]

        text_seq_tokens = self.text_tokens_fc(text_seq_tokens)  # [batch_size, 40, 512]
        image_patch_tokens = self.image_tokens_fc(
            image_patch_tokens
        )  # [batch_size, 50, 512]

        text_cls = clip_output.text_embeds  # [batch_size, 512]
        image_cls = clip_output.image_embeds  # [batch_size, 512]

        return text_seq_tokens, image_patch_tokens, text_cls, image_cls


class OT_Transport(nn.Module):
    def __init__(self, args, input_dim, hidden_dim):
        super(OT_Transport, self).__init__()
        self.args = args

        self.trans_ori = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.trans_tar = nn.Linear(in_features=input_dim, out_features=hidden_dim)

        self.fc_origin = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.fc_origin_transport = nn.Linear(
            in_features=hidden_dim, out_features=hidden_dim
        )
        self.fc_target = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=self.args.model.dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.mse = nn.MSELoss()

    def forward(self, origin_seq, target_seq):
        origin_seq = self.trans_ori(
            origin_seq
        )  # [batch_size, origin_seq_len, hidden_dim]
        target_seq = self.trans_tar(
            target_seq
        )  # [batch_size, target_seq_len, hidden_dim]

        target_seq = self.fc_target(
            target_seq
        )  # [batch_size, target_seq_len, hidden_dim]
        origin_seq_fc = self.fc_origin(
            origin_seq
        )  # [batch_size, origin_seq_len, hidden_dim]
        origin_seq_transport = self.fc_origin_transport(
            origin_seq
        )  # [batch_size, origin_seq_len, hidden_dim]

        # [batch_size_a, seq_len_a, seq_len_b]
        C = cosine_sim(target_seq, origin_seq_fc)
        transport_plan = sinkhorn_torch(
            C=C,
            epsilon=self.args.model.ot_reg,
            sinkhorn_iterations=self.args.model.ot_transport,
        )
        transport_plan = F.normalize(input=transport_plan, dim=-1)
        transport_plan = self.dropout(transport_plan)

        transport_seq = self.layernorm(
            torch.matmul(transport_plan, origin_seq_transport) + target_seq
        )  # [batch_size, target_seq, hidden_dim]

        return transport_seq


class CrossAttLocalBatch(nn.Module):
    def __init__(self, args, input_dim, hidden_dim):
        super(CrossAttLocalBatch, self).__init__()
        self.fc_q = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.fc_k = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.fc_v = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.mse = nn.MSELoss()
        self.args = args
        self.dropout = nn.Dropout(p=self.args.model.dropout)

    def forward(self, batch_a, batch_b):
        # batch_a: [batch_size_a, seq_len_a, dim]
        # batch_b: [batch_size_b, seq_len_b, dim]
        query = self.fc_q(batch_a)
        key = self.fc_k(batch_b)
        value = self.fc_v(batch_b)

        batch_size_a, seq_len_a, dim = query.shape
        batch_size_b, seq_len_b, _ = key.shape

        # batch_a_expanded: [batch_size_a, 1, seq_len_a, dim]
        # batch_b_expanded: [1, batch_size_b, seq_len_b, dim]
        query_expanded = query.unsqueeze(1).expand(-1, batch_size_b, -1, -1)
        key_expanded = key.unsqueeze(0).expand(batch_size_a, -1, -1, -1)

        # attention_scores: [batch_size_a, batch_size_b, seq_len_a, seq_len_b]
        C = cosine_sim(query_expanded, key_expanded)

        transport_plan = sinkhorn_torch(
            C=C.detach(),
            epsilon=self.args.model.ot_reg,
            sinkhorn_iterations=self.args.model.ot_transport,
        )
        # attention_weights: [batch_size_a, batch_size_b, seq_len_a, seq_len_b]
        transport_plan = F.normalize(input=transport_plan, dim=-1)
        transport_plan = self.dropout(transport_plan)

        # attended_values: [batch_size_a, batch_size_b, seq_len_a, dim]
        attended_values = self.layernorm(
            torch.einsum("abij,bjd->abid", transport_plan, value) + query_expanded
        )

        return attended_values


class SoftPoolModule(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(SoftPoolModule, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, x):
        weights = self.softmax(self.fc(x))  # Shape: (batch_size, seq_length, 1)

        weighted_sum = torch.sum(x * weights, dim=-2)  # Shape: (batch_size, input_dim)
        normalized_output = self.layer_norm(weighted_sum)

        return normalized_output


class MultimodalMatcher(nn.Module):
    def __init__(self, args, input_dim, hidden_dim):
        super(MultimodalMatcher, self).__init__()
        self.args = args

        self.image2text_transport = OT_Transport(args, input_dim, hidden_dim)
        self.text2image_transport = OT_Transport(args, input_dim, hidden_dim)

        self.soft_pool_0 = SoftPoolModule(input_dim=self.args.model.hidden_dim)
        self.soft_pool_1 = SoftPoolModule(input_dim=self.args.model.hidden_dim)

        self.fc_text = nn.Linear(input_dim, hidden_dim)
        self.fc_image = nn.Linear(input_dim, hidden_dim)

    def forward(
        self,
        entity_text_tokens,
        mention_text_tokens,
        entity_image_tokens,
        mention_image_tokens,
    ):
        # Entity Transport
        entity_image_transported = self.image2text_transport(
            origin_seq=entity_image_tokens,
            target_seq=entity_text_tokens,
        )  # [num_entity, text_seq_len, dim]
        entity_text_transported = self.text2image_transport(
            origin_seq=entity_text_tokens,
            target_seq=entity_image_tokens,
        )  # [num_entity, num_patch, dim]

        # Sequence Pooling
        entity_multimodal_seq_0 = self.soft_pool_0(
            torch.cat(
                [self.fc_text(entity_text_tokens), entity_image_transported], dim=1
            )
        )  # [num_entity, dim]
        entity_multimodal_seq_1 = self.soft_pool_1(
            torch.cat(
                [entity_text_transported, self.fc_image(entity_image_tokens)], dim=1
            )
        )  # [num_entity, dim]

        # Mention Transport
        mention_image_transported = self.image2text_transport(
            origin_seq=mention_image_tokens,
            target_seq=mention_text_tokens,
        )  # [num_entity, text_seq_len, dim]
        mention_text_transported = self.text2image_transport(
            origin_seq=mention_text_tokens,
            target_seq=mention_image_tokens,
        )  # [num_entity, num_patch, dim]

        # Sequence Pooling
        mention_multimodal_seq_0 = self.soft_pool_0(
            torch.cat(
                [self.fc_text(mention_text_tokens), mention_image_transported],
                dim=1,
            )
        )  # [batch_size, dim]
        mention_multimodal_seq_1 = self.soft_pool_1(
            torch.cat(
                [mention_text_transported, self.fc_image(mention_image_tokens)],
                dim=1,
            )
        )  # [batch_size, dim]

        # Multimodal Local Matching
        multimodal_local_matching_score_0 = torch.matmul(
            mention_multimodal_seq_0, entity_multimodal_seq_0.T
        )
        multimodal_local_matching_score_1 = torch.matmul(
            mention_multimodal_seq_1, entity_multimodal_seq_1.T
        )

        multimodal_matching_score = (
            multimodal_local_matching_score_0 + multimodal_local_matching_score_1
        ) / 2

        return multimodal_matching_score


class UniModelMatcher(nn.Module):
    def __init__(self, args):
        super(UniModelMatcher, self).__init__()
        self.args = args

        self.cross = CrossAttLocalBatch(
            args, self.args.model.input_hidden_dim, self.args.model.hidden_dim
        )

        self.fc_cls = nn.Linear(
            self.args.model.input_hidden_dim, self.args.model.hidden_dim
        )

        self.softpool = SoftPoolModule(input_dim=self.args.model.hidden_dim)

    def forward(self, entity_cls, entity_tokens, mention_cls, mention_tokens):
        """

        :param entity_cls:     [num_entity, dim]
        :param entity_tokens:  [num_entity, max_seq_len, dim]
        :param mention_cls:    [batch_size, dim]
        :param mention_tokens: [batch_size, max_sqe_len, dim]
        :return:
        """

        entity_cls_fc = self.fc_cls(entity_cls)  # [num_entity, dim]
        entity_cls_fc = entity_cls_fc.unsqueeze(dim=1)  # [num_entity, 1, dim]

        context = self.cross(
            entity_tokens, mention_tokens
        )  # [num_entity, batch_size, max_seq_len, dim]

        context = self.softpool(context)  # [num_entity, batch_size, dim]

        g2l_matching_score = torch.sum(
            entity_cls_fc * context, dim=-1
        )  # [num_entity, batch_size]
        g2l_matching_score = g2l_matching_score.transpose(
            0, 1
        )  # [batch_size, num_entity]
        g2g_matching_score = torch.matmul(mention_cls, entity_cls.transpose(-1, -2))

        matching_score = (g2l_matching_score + g2g_matching_score) / 2

        return matching_score


class Matcher(nn.Module):
    def __init__(self, args):
        super(Matcher, self).__init__()
        self.args = args

        self.text_tokens_layernorm = nn.LayerNorm(self.args.model.input_hidden_dim)
        self.image_tokens_layernorm = nn.LayerNorm(self.args.model.input_hidden_dim)
        self.text_cls_layernorm = nn.LayerNorm(self.args.model.input_hidden_dim)
        self.image_cls_layernorm = nn.LayerNorm(self.args.model.input_hidden_dim)

        self.multimodal_matcher = MultimodalMatcher(
            args=args,
            input_dim=self.args.model.input_hidden_dim,
            hidden_dim=self.args.model.hidden_dim,
        )

        self.text_matcher = UniModelMatcher(args=args)
        self.image_matcher = UniModelMatcher(args=args)

    def forward(
        self,
        entity_text_tokens,
        mention_text_tokens,
        entity_image_tokens,
        mention_image_tokens,
        entity_text_cls,
        mention_text_cls,
        entity_image_cls,
        mention_image_cls,
    ):
        """
        :param entity_text_tokens:  [num_entity, text_seq_len, dim]
        :param mention_text_tokens: [batch_size, text_seq_len, dim]
        :param entity_image_tokens: [num_entity, num_patch, dim]
        :param mention_image_tokens:[batch_size, num_patch, dim]
        :return:
        """
        entity_text_tokens = self.text_tokens_layernorm(
            entity_text_tokens
        )  # [batch_size, seq_len, dim]
        mention_text_tokens = self.text_tokens_layernorm(
            mention_text_tokens
        )  # [num_entity, seq_len, dim]

        entity_image_tokens = self.image_tokens_layernorm(
            entity_image_tokens
        )  # [num_entity, num_patch, dim]
        mention_image_tokens = self.image_tokens_layernorm(
            mention_image_tokens
        )  # [batch_size, num_patch, dim]

        mention_text_cls = self.text_cls_layernorm(mention_text_cls)
        entity_text_cls = self.text_cls_layernorm(entity_text_cls)
        mention_image_cls = self.image_cls_layernorm(mention_image_cls)
        entity_image_cls = self.image_cls_layernorm(entity_image_cls)

        # Text Matching
        text_matching_score = self.text_matcher(
            entity_text_cls,
            entity_text_tokens,
            mention_text_cls,
            mention_text_tokens,
        )  # [batch_size, num_entity]

        # Image Matching
        image_matching_score = self.image_matcher(
            entity_image_cls,
            entity_image_tokens,
            mention_image_cls,
            mention_image_tokens,
        )  # [batch_size, num_entity]

        # Multimodal Local Matching
        multimodal_matching_score = self.multimodal_matcher(
            entity_text_tokens,
            mention_text_tokens,
            entity_image_tokens,
            mention_image_tokens,
        )  # [batch_size, num_entity]

        matching_score = (
            text_matching_score + image_matching_score + multimodal_matching_score
        ) / 3

        return (
            matching_score,
            (text_matching_score, image_matching_score, multimodal_matching_score),
        )
