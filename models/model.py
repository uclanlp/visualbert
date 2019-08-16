# Modified from VCR.

from typing import Dict, List, Any
import os

import torch
import torch.nn.functional as F
import torch.nn as nn

import torch.nn.parallel
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, FeedForward, InputVariationalDropout, TimeDistributed
from allennlp.training.metrics import CategoricalAccuracy, Average
from allennlp.modules.matrix_attention import BilinearMatrixAttention

from allennlp.nn.util import masked_softmax, weighted_sum, replace_masked_values
from allennlp.nn import InitializerApplicator

from pytorch_pretrained_bert.modeling import BertForMultipleChoice, TrainVisualBERTObjective #BertForMultipleChoice, BertForVisualMultipleChoice, BertForVisualPreTraining, BertForPreTraining, BertForVisualQA
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

@Model.register("VisualBERTDetector")
class VisualBERTDetector(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 class_embs: bool=True,
                 bert_model_name: str="bert-base-uncased",
                 cnn_loss_ratio: float=0.0,
                 special_visual_initialize: bool=False,
                 text_only: bool=False,
                 visual_embedding_dim: int=512,
                 hard_cap_seq_len: int=None,
                 cut_first: str='text',
                 embedding_strategy: str='plain',
                 random_initialize: bool=False,
                 training_head_type: str="pretraining",
                 bypass_transformer: bool=False,
                 pretrained_detector: bool=True,
                 output_attention_weights: bool=False
                 ):
        super(VisualBERTDetector, self).__init__(vocab)
        
        from utils.detector import SimpleDetector
        self.detector = SimpleDetector(pretrained=pretrained_detector, average_pool=True, semantic=class_embs, final_dim=512)
        ##################################################################################################
        self.bert = TrainVisualBERTObjective.from_pretrained(
                bert_model_name,
                cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(-1)),
                training_head_type = training_head_type,
                visual_embedding_dim = visual_embedding_dim,
                hard_cap_seq_len = hard_cap_seq_len,
                cut_first = cut_first,
                embedding_strategy = embedding_strategy,
                bypass_transformer = bypass_transformer,
                random_initialize = random_initialize,
                output_attention_weights = output_attention_weights)
        if special_visual_initialize:
            self.bert.bert.embeddings.special_intialize()
    
        self.training_head_type = training_head_type
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        self.cnn_loss_ratio = cnn_loss_ratio

    def _collect_obj_reps(self, span_tags, object_reps):
        """
        Collect span-level object representations
        :param span_tags: [batch_size, ..leading_dims.., L]
        :param object_reps: [batch_size, max_num_objs_per_batch, obj_dim]
        :return:
        """
        span_tags_fixed = torch.clamp(span_tags, min=0)  # In case there were masked values here
        row_id = span_tags_fixed.new_zeros(span_tags_fixed.shape)
        row_id_broadcaster = torch.arange(0, row_id.shape[0], step=1, device=row_id.device)[:, None]

        # Add extra diminsions to the row broadcaster so it matches row_id
        leading_dims = len(span_tags.shape) - 2
        for i in range(leading_dims):
            row_id_broadcaster = row_id_broadcaster[..., None]
        row_id += row_id_broadcaster
        return object_reps[row_id.view(-1), span_tags_fixed.view(-1)].view(*span_tags_fixed.shape, -1)

    def embed_span(self, span, span_tags, span_mask, object_reps):
        """
        :param span: Thing that will get embed and turned into [batch_size, ..leading_dims.., L, word_dim]
        :param span_tags: [batch_size, ..leading_dims.., L]
        :param object_reps: [batch_size, max_num_objs_per_batch, obj_dim]
        :param span_mask: [batch_size, ..leading_dims.., span_mask
        :return:
        """
        retrieved_feats = self._collect_obj_reps(span_tags, object_reps)

        span_rep = torch.cat((span['bert'], retrieved_feats), -1)
        # add recurrent dropout here
        if self.rnn_input_dropout:
            span_rep = self.rnn_input_dropout(span_rep)

        return self.span_encoder(span_rep, span_mask), retrieved_feats

    def forward(self,
                images: torch.Tensor = None,
                objects: torch.LongTensor = None,
                segms: torch.Tensor = None,
                boxes: torch.Tensor = None,
                box_mask: torch.LongTensor = None,
                question: Dict[str, torch.Tensor] = None,
                question_tags: torch.LongTensor = None,
                question_mask: torch.LongTensor = None,
                answers: Dict[str, torch.Tensor] = None,
                answer_tags: torch.LongTensor = None,
                answer_mask: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None,
                label: torch.LongTensor = None,
                bert_input_ids: torch.LongTensor = None,
                bert_input_mask: torch.LongTensor = None,
                bert_input_type_ids: torch.LongTensor = None,
                masked_lm_labels: torch.LongTensor = None,
                is_random_next: torch.LongTensor= None,
                image_text_alignment: torch.LongTensor = None,
                output_all_encoded_layers = False) -> Dict[str, torch.Tensor]:

        # Trim off boxes that are too long. this is an issue b/c dataparallel, it'll pad more zeros that are
        # not needed

        max_len = int(box_mask.sum(1).max().item())
        objects = objects[:, :max_len]
        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]
        segms = segms[:, :max_len]
        '''for tag_type, the_tags in (('question', question_tags), ('answer', answer_tags)):
            if int(the_tags.max()) > max_len:
                raise ValueError("Oh no! {}_tags has maximum of {} but objects is of dim {}. Values are\n{}".format(
                    tag_type, int(the_tags.max()), objects.shape, the_tags
                ))'''
        obj_reps = self.detector(images=images, boxes=boxes, box_mask=box_mask, classes=objects, segms=segms)

        #print("obj_reps", obj_reps['obj_reps'].size())
        #print("bert_input_ids", bert_input_ids.size())
        #print("box_mask", box_mask.size())

        if len(bert_input_ids.size()) == 2: # Using complete shuffle mode
            obj_reps_expanded = obj_reps['obj_reps']
            box_mask_expanded = box_mask
        else:
            obj_reps_expanded = obj_reps['obj_reps'].unsqueeze(1).expand(box_mask.size(0), bert_input_mask.size(1), box_mask.size(-1), obj_reps['obj_reps'].size(-1))
            box_mask_expanded = box_mask.unsqueeze(1).expand(box_mask.size(0), bert_input_mask.size(1), box_mask.size(-1))
        
        #bert_input_mask = torch.cat((bert_input_mask, box_mask_expanded), dim = -1)

        output_dict = self.bert(
            input_ids = bert_input_ids, 
            token_type_ids = bert_input_type_ids, 
            input_mask = bert_input_mask, 

            visual_embeddings = obj_reps_expanded, 
            position_embeddings_visual = None,
            image_mask = box_mask_expanded,
            visual_embeddings_type = None,

            image_text_alignment = image_text_alignment,

            label = label,
            masked_lm_labels = masked_lm_labels,
            is_random_next = is_random_next,

            output_all_encoded_layers = output_all_encoded_layers)

        #class_probabilities = F.softmax(logits, dim=-1)
        cnn_loss = obj_reps['cnn_regularization_loss']
        if self.cnn_loss_ratio == 0.0:
            output_dict["cnn_regularization_loss"] = None
        else:
            output_dict["cnn_regularization_loss"] = cnn_loss * self.cnn_loss_ratio

        # Multi-process safe??
        if label is not None and self.training_head_type != "pretraining":
            logits = output_dict["logits"]
            logits = logits.detach().float()
            label = label.float()
            self._accuracy(logits, label)

        if self.training_head_type == "pretraining":
            output_dict["logits"] = None # Because every image may has different number of image features, the lengths of the logits on different GPUs will be different. This will cause DataParallel to throw errors.

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self._accuracy.get_metric(reset)}

@Model.register("VisualBERTFixedImageEmbedding")
class VisualBERTFixedImageEmbedding(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 class_embs: bool=True,
                 bert_model_name: str="bert-base-uncased",
                 cnn_loss_ratio: float=0.0,
                 special_visual_initialize: bool=False,
                 text_only: bool=False,
                 training_head_type: str='',
                 visual_embedding_dim: int=512,
                 hard_cap_seq_len: int=None,
                 cut_first: str='text',
                 embedding_strategy: str='plain',
                 random_initialize: bool=False,
                 bypass_transformer: bool=False,
                 output_attention_weights: bool=False
                 ):
        super(VisualBERTFixedImageEmbedding, self).__init__(vocab)
        self.text_only = text_only
        self.training_head_type = training_head_type

        self.bert = TrainVisualBERTObjective.from_pretrained(
                bert_model_name,
                cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(-1)),
                training_head_type = training_head_type,
                visual_embedding_dim = visual_embedding_dim,
                hard_cap_seq_len = hard_cap_seq_len,
                cut_first = cut_first,
                embedding_strategy = embedding_strategy,
                bypass_transformer = bypass_transformer,
                random_initialize = random_initialize,
                output_attention_weights = output_attention_weights)
        if special_visual_initialize:
            self.bert.bert.embeddings.special_intialize()

        if self.training_head_type == "nlvr" or self.training_head_type == "multichoice":
            self._accuracy = CategoricalAccuracy()
        if "vqa" in self.training_head_type:
            self._accuracy = Average()
        if self.training_head_type == "flickr":
            self._accuracy = Average()

    def forward(self,

                #bert text input
                bert_input_ids,
                bert_input_mask,
                bert_input_type_ids,

                # image input
                image_dim_variable = None,
                image_feat_variable = None,

                #
                image_text_alignment = None,
                visual_embeddings_type = None,

                # fine-tuning label
                label = None,
                flickr_position = None, # For flickr we also need to provide the position

                # pretraining lables
                masked_lm_labels = None,
                is_random_next = None,

                output_all_encoded_layers = False
                ) -> Dict[str, torch.Tensor]:

        # image_feat_variable = batch x ( num_choice x ) image_feature_length x dim
        # Prepare Mask
        if image_feat_variable is not None:
            image_mask = torch.arange(image_feat_variable.size(-2)).expand(*image_feat_variable.size()[:-1]).cuda() 
            if len(image_dim_variable.size()) < len(image_mask.size()):
                image_dim_variable = image_dim_variable.unsqueeze(-1)
                assert(len(image_dim_variable.size()) == len(image_mask.size()))
            image_mask = image_mask < image_dim_variable
            image_mask = image_mask.long()
        else:
            image_mask = None

        output_dict = self.bert(
            input_ids = bert_input_ids, 
            token_type_ids = bert_input_type_ids, 
            input_mask = bert_input_mask, 

            visual_embeddings = image_feat_variable, 
            position_embeddings_visual = None,
            image_mask = image_mask,
            visual_embeddings_type = visual_embeddings_type,
            image_text_alignment = image_text_alignment,

            label = label,
            flickr_position = flickr_position,
            masked_lm_labels = masked_lm_labels,
            is_random_next = is_random_next,

            output_all_encoded_layers = output_all_encoded_layers)

        if self.training_head_type == "nlvr" or self.training_head_type == "multichoice":
            logits = output_dict["logits"]
            self._accuracy(logits, label)

        # Multi-process safe??
        if "vqa" in self.training_head_type or self.training_head_type == "flickr":
            if output_dict["accuracy"] is not None:
                self._accuracy(output_dict["accuracy"])

        output_dict["cnn_regularization_loss"] = None

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        if self.training_head_type == "nlvr" or self.training_head_type == "multichoice" or "vqa" in self.training_head_type or self.training_head_type == "flickr":
            return {'accuracy': self._accuracy.get_metric(reset)}
        return {'accuracy': 0.0}

    @staticmethod
    def compute_score_with_logits(logits, labels):
        logits = masked_unk_softmax(logits, 1, 0)
        logits = torch.max(logits, 1)[1].data  # argmax
        one_hots = torch.zeros(*labels.size())
        one_hots = one_hots.cuda() if use_cuda else one_hots
        one_hots.scatter_(1, logits.view(-1, 1), 1)
        scores = (one_hots * labels)
        return scores

class SimpleReportMetric():
    def __init__(self):
        self.total = 0.0
        self.called_time = 0

    def __call__(self, number, *args):
        if isinstance(number, torch.Tensor):
            number = number.item()
        self.total += number
        self.called_time += 1

    def get_metric(self, reset):
        return

