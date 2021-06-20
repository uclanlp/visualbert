# coding=utf-8
# Copyleft 2019 project LXRT.

import torch.nn as nn

from param import args
from lxrt.entry import LXRTEncoder, convert_sents_to_features_tensors, convert_tags_to_tensorts, pad_np_arrays
from lxrt.modeling import BertLayerNorm, GeLU
from lxrt.tokenization import BertTokenizer
import numpy as np

# Max length including <bos> and <eos>
MAX_VQA_LENGTH = 20


class VQAModel(nn.Module):
    def __init__(self, num_answers):
        super().__init__()
        
        # Build LXRT encoder
        self.lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=MAX_VQA_LENGTH
        )
        hid_dim = self.lxrt_encoder.dim
        
        # VQA Answer heads
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        )
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)

        self.tokenizer = BertTokenizer.from_pretrained(
                "bert-base-uncased",
                do_lower_case=True
            )
    
    def multi_gpu(self):
        self.lxrt_encoder.model.module.bert = nn.DataParallel(self.lxrt_encoder.model.module.bert)

    def forward(self, feat, pos, sent, tags):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        #x = self.lxrt_encoder(sent, (feat, pos))
        input_ids, input_mask, segment_ids = convert_sents_to_features_tensors(sent, max_seq_length = MAX_VQA_LENGTH, tokenizer=self.tokenizer)
        
        visual_tags, visual_tags_mask, visual_tags_box, visual_tags_type, visual_tags_segment_ids = convert_tags_to_tensorts(tags)

        feat = pad_np_arrays(feat, padding_value=0, dtype=np.float32)
        pos = pad_np_arrays(pos, padding_value=0, dtype=np.float32)

        stuff, pooled_output = self.lxrt_encoder.model.module.bert(
                input_ids, segment_ids, input_mask,
                visual_feats=(feat, pos),
                visual_attention_mask=None,
                visual_feats_seg_ids=None,
                visual_tags=visual_tags, visual_tags_mask=visual_tags_mask, visual_tags_box=visual_tags_box, visual_tags_type=visual_tags_type, visual_tags_segment_ids=visual_tags_segment_ids,
                )
        logit = self.logit_fc(pooled_output)

        return logit


