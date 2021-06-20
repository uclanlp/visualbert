import os
from torch.utils.data import Dataset
import numpy as np

import random

import json
from collections import defaultdict
from tqdm import tqdm
import json
import os

import numpy as np
import numpy
import torch
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField, ListField, LabelField, SequenceLabelField, ArrayField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.util import get_text_field_mask
from torch.utils.data import Dataset
from dataloaders.box_utils import load_image, resize_image, to_tensor_and_normalize
from dataloaders.mask_utils import make_mask
from dataloaders.bert_field import BertField
import h5py
from copy import deepcopy

from torch.utils.data.dataloader import default_collate
from allennlp.data.instance import Instance
from allennlp.data.dataset import Batch
from pytorch_pretrained_bert.fine_tuning import _truncate_seq_pair, random_word
from dataloaders.bert_field import IntArrayField
import numpy as np
from allennlp.data.fields import ListField
from visualbert.pytorch_pretrained_bert.tokenization import BertTokenizer

from .bert_data_utils import *

class NLVRDataset(Dataset):
    def __init__(self, args):
        super(NLVRDataset, self).__init__()
        self.args = args
        self.annots_path = args.annots_path
        self.split = args.split
        self.text_only = args.get("text_only", False)
        self.no_next_sentence = args.get("no_next_sentence", False)
        
        with open(self.annots_path, 'r') as f:
            self.items = [json.loads(s) for s in f]

        self.image_feat_reader = faster_RCNN_feat_reader()
        self.image_screening_parameters = self.args.image_screening_parameters
        if args.get("chunk_path", None) is not None:
            self.chunk = torch.load(args.chunk_path)
            average = 0.0
            new_chunk = {}
            for image_id in self.chunk.keys():
                image_feat_variable, image_boxes, confidence  = self.chunk[image_id]
                if "npz" not in image_id:
                    new_chunk[image_id+".npz"] = screen_feature(image_feat_variable, image_boxes,confidence, self.image_screening_parameters)
                    average += new_chunk[image_id+".npz"][2]
                else:
                    new_chunk[image_id] = screen_feature(image_feat_variable, image_boxes,confidence, self.image_screening_parameters)
                    average += new_chunk[image_id][2]
            self.chunk = new_chunk
            print("{} features on average.".format(average/len(self.chunk)))

        ##########
        self.do_lower_case = args.do_lower_case
        self.bert_model_name = args.bert_model_name

        self.max_seq_length = args.max_seq_length
        # 1. Initialize the BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model_name, do_lower_case=self.do_lower_case)
        self.pretraining = args.pretraining

        # This is for pretraining
        self.masked_lm_prob = 0.15
        self.max_predictions_per_seq = 20

    def get_image_features_by_training_index(self, index, which_one):
        item = self.items[index]

        image_file_name = "{}img{}.png.npz".format(item['identifier'][:-1], which_one)

        return self.chunk[image_file_name]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        sample = {}

        if not self.text_only:
            image_feat_variable_0, image_boxes_0, image_dim_variable_0 = self.get_image_features_by_training_index(index, 0)
            image_feat_variable_1, image_boxes_1, image_dim_variable_1 = self.get_image_features_by_training_index(index, 1)

            visual_embeddings_type_0 = np.zeros(image_feat_variable_0.shape[0])
            visual_embeddings_type_1 = np.ones(image_feat_variable_1.shape[0])

            visual_embeddings_type = numpy.concatenate((visual_embeddings_type_0, visual_embeddings_type_1), axis = 0)
            image_feat_variable = numpy.concatenate((image_feat_variable_0, image_feat_variable_1), axis = 0)
            image_dim_variable = image_dim_variable_0 + image_dim_variable_1
            image_feat_variable = torch.Tensor(image_feat_variable)
            #image_boxes = ArrayField(image_boxes)
            image_dim_variable = torch.LongTensor([image_dim_variable])
            visual_embeddings_type = torch.LongTensor(visual_embeddings_type)
            sample["image_feat_variable"] = image_feat_variable
            #sample["image_boxes"] = image_boxes
            sample["image_dim_variable"] = image_dim_variable
            sample["visual_embeddings_type"] = visual_embeddings_type

        caption_a = item["sentence"]

        if item.get("label", None) is not None:
            sample["label"] = torch.LongTensor([1 if item["label"] == "True" else 0])
        else:
            sample["label"] = torch.LongTensor([0]) # Pseudo label

        if self.pretraining:
            subword_tokens_a = self.tokenizer.tokenize(caption_a)
            if self.no_next_sentence:
                bert_example = InputExample(unique_id = index, text_a = subword_tokens_a, text_b = None, is_correct=None, max_seq_length = self.max_seq_length)
            else:
                bert_example = InputExample(unique_id = index, text_a = subword_tokens_a, text_b = None, is_correct=1 if item["label"] == "True" else 0, max_seq_length = self.max_seq_length)
            bert_feature = InputFeatures.convert_one_example_to_features_pretraining(
                        example = bert_example,
                        tokenizer=self.tokenizer,
                        probability = 0.15)
            bert_feature.insert_tensor_into_dict(sample)
        else:
            subword_tokens_a = self.tokenizer.tokenize(caption_a)
            bert_example = InputExample(unique_id = index, text_a = subword_tokens_a, text_b = None, is_correct=1 if item.get("label", None) == "True" else 0, max_seq_length = self.max_seq_length)
            bert_feature = InputFeatures.convert_one_example_to_features(
                example = bert_example,
                tokenizer=self.tokenizer)
            bert_feature.insert_tensor_into_dict(sample)

        return sample


    @classmethod
    def splits(cls, args):
        data_root = args.data_root
        args_copy = deepcopy(args)
        args_copy.split = "train"
        args_copy.annots_path = os.path.join(data_root, "{}.json".format(args_copy.split))

        if args.image_screening_parameters["image_feature_cap"] is not None and args.image_screening_parameters["image_feature_cap"] > 36:
            args_copy.chunk_path = os.path.join(data_root, "features_{}_150.th".format(args_copy.split))
        else:
            args_copy.chunk_path = os.path.join(data_root, "features_chunk_{}.th".format(args_copy.split))

        if args.get("do_test", False):
            trainset = None
            validationset = None
        else:
            trainset = cls(args_copy)
            trainset.is_train = True
            args_copy = deepcopy(args)
            args_copy.split = "dev"
            args_copy.annots_path = os.path.join(data_root, "{}.json".format(args_copy.split))
            if args.image_screening_parameters["image_feature_cap"] is not None and args.image_screening_parameters["image_feature_cap"] > 36:
                args_copy.chunk_path = os.path.join(data_root, "features_{}_150.th".format(args_copy.split))
            else:
                args_copy.chunk_path = os.path.join(data_root, "features_chunk_{}.th".format(args_copy.split))

            validationset = cls(args_copy)
            validationset.is_train = False

        args_copy = deepcopy(args)
        args_copy.split = "test1"

        if args.get("test_on_hidden", False):
            args_copy.split = "test2"

        args_copy.annots_path = os.path.join(data_root, "{}.json".format(args_copy.split))
        if args.image_screening_parameters["image_feature_cap"] is not None and args.image_screening_parameters["image_feature_cap"] > 36:
            args_copy.chunk_path = os.path.join(data_root, "features_{}_150.th".format(args_copy.split))
        else:
            args_copy.chunk_path = os.path.join(data_root, "features_chunk_{}.th".format(args_copy.split))
        testset = cls(args_copy)
        testset.is_train = False

        if args.get("do_test", False):
            trainset = testset
            validationset = testset

        return trainset, validationset, testset

    @staticmethod
    def collate_fn(data):
        if isinstance(data[0], dict):
            for index, i in enumerate(data):
                if "image_feat_variable" in i:
                    i["image_feat_variable"] = ArrayTensorField(i["image_feat_variable"])
                    i["image_dim_variable"] = IntArrayTensorField(i["image_dim_variable"])
                    i["visual_embeddings_type"] = IntArrayTensorField(i["visual_embeddings_type"])

                i["bert_input_ids"] = IntArrayTensorField(i["bert_input_ids"])
                i["bert_input_mask"] = IntArrayTensorField(i["bert_input_mask"])
                i["bert_input_type_ids"] = IntArrayTensorField(i["bert_input_type_ids"])

                if "masked_lm_labels" in i:
                    i["masked_lm_labels"] = IntArrayTensorField(i["masked_lm_labels"], padding_value = -1)
                if "is_random_next" in i:
                    i["is_random_next"] = IntArrayTensorField(i["is_random_next"])
                i['label'] = IntArrayTensorField(i['label'])

                data[index] = Instance(i)
        batch = Batch(data)
        td = batch.as_tensor_dict()
        td["label"] = td["label"].squeeze(-1)
        return td