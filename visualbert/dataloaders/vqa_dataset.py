# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from torch.utils.data import Dataset
import numpy as np
from copy import deepcopy

import torch
from torch.utils.data.dataloader import default_collate
from allennlp.data.instance import Instance
from allennlp.data.dataset import Batch

from allennlp.data.fields import TextField, ListField, LabelField, SequenceLabelField, ArrayField, MetadataField
from dataloaders.bert_field import IntArrayField

from .bert_data_utils import *
from visualbert.pytorch_pretrained_bert.tokenization import BertTokenizer

imdb_version = 1 # Not sure what this does... Just follow it

import random
import json

def compute_answer_scores(answers, num_of_answers, unk_idx):
    scores = np.zeros((num_of_answers), np.float32)
    for answer in set(answers):
        if answer == unk_idx:
            scores[answer] = 0
        else:
            answer_count = answers.count(answer)
            scores[answer] = min(np.float32(answer_count)*0.3, 1)
    return scores


def read_in_image_feats(image_dirs, image_readers, image_file_name):
    image_feats = []
    for i, image_dir in enumerate(image_dirs):
        image_feat_path = os.path.join(image_dir, image_file_name)
        tmp_image_feat = image_readers[i].read(image_feat_path)
        image_feats.append(tmp_image_feat)

    return image_feats



class VQADataset(Dataset):
    def __init__(self, args, chunk_train = None, chunk_val = None): # Using args is not exactly a very good coding habit...
        super(VQADataset, self).__init__()

        if isinstance(args.imdb_file, list) or isinstance(args.imdb_file, tuple): # For training dataset, the imdb_file is a list of strs, containing train and val:
            imdb = np.load(args.imdb_file[0], allow_pickle = True)[1:]
            for i in args.imdb_file[1:]:
                imdb_i = np.load(i, allow_pickle = True)[1:]
                imdb = np.concatenate((imdb, imdb_i))
        else:
            if args.imdb_file.endswith('.npy'):
                imdb = np.load(args.imdb_file, allow_pickle = True)[1:]
            else:
                raise TypeError('unknown imdb format.')

        self.items = imdb
        self.chunk_train = chunk_train
        self.chunk_val = chunk_val

        self.args = args
        self.data_root = args.data_root
        self.use_visual_genome = args.use_visual_genome
        self.pretraining = args.pretraining
        self.include_res152 = args.get('include_res152', False)
        self.no_next_sentence = args.get("no_next_sentence", False)
        self.false_caption_ratio = args.get("false_caption_ratio", 0.5)

        # the answer dict is always loaded, regardless of self.load_answer
        self.answer_dict = VocabDict(args.vocab_answer_file)
        self.do_lower_case = args["do_lower_case"]
        self.bert_model_name = args.bert_model_name
        self.max_seq_length = args.max_seq_length
        self.pretraining = args.pretraining

        self.masked_lm_prob = args.get("masked_lm_prob", 0.15)
        self.tokenizer = BertTokenizer.from_pretrained(args["bert_model_name"], do_lower_case=args["do_lower_case"])

        self.advanced_vqa = True if args.model.training_head_type == "vqa_advanced" else False
        if self.advanced_vqa:
            tokenized_list = []
            for i in self.answer_dict.word_list:
                tokenized_list.append(self.tokenizer.tokenize(i))
            max_len = max(len(i) for i in tokenized_list)
            for index, i in enumerate(tokenized_list):
                if len(i) < max_len:
                    tokenized_list[index] = i + ["[MASK]"] * (max_len - len(i))
            self.tokenized_list = tokenized_list

    def __len__(self):
        return len(self.items)

    def get_image_features_by_training_index(self, index):

        if not self.use_visual_genome:
            item = self.items[index]
            image_file_name = self.items[index]['image_name'] + ".jpg.npz"
            try:
                return self.chunk_train[image_file_name]
            except:
                return self.chunk_val[image_file_name]
        else:
            iminfo = self.items[index]
            image_file_name = iminfo['image_name'] + ".npy"
            if "train" in image_file_name:
                folder = os.path.join(self.data_root, "data/detectron_fix_100/fc6/vqa/train2014")
            elif "val" in image_file_name:
                folder = os.path.join(self.data_root, "data/detectron_fix_100/fc6/vqa/val2014")
            elif "test" in image_file_name:
                folder = os.path.join(self.data_root, "data/detectron_fix_100/fc6/vqa/test2015")
            detectron_features = np.load(os.path.join(folder, image_file_name))
            image_feat_variable = detectron_features
            image_dim_variable = image_feat_variable.shape[0]
            visual_embeddings_type = None
            return  image_feat_variable, None, image_dim_variable

    def __getitem__(self, index):

        iminfo = self.items[index] 
        image_feat_variable, image_boxes, image_dim_variable = self.get_image_features_by_training_index(index)

        sample = {}

        image_feat_variable = ArrayField(image_feat_variable)
        image_dim_variable = IntArrayField(np.array(image_dim_variable))
        sample["image_feat_variable"] = image_feat_variable
        sample["image_dim_variable"] = image_dim_variable

        answer = None
        valid_answers_idx = np.zeros((10), np.int32)
        valid_answers_idx.fill(-1)
        answer_scores = np.zeros(self.answer_dict.num_vocab, np.float32)

        if 'answer' in iminfo:
            answer = iminfo['answer']
        elif 'valid_answers' in iminfo:
            valid_answers = iminfo['valid_answers']
            answer = np.random.choice(valid_answers)
            valid_answers_idx[:len(valid_answers)] = (
                [self.answer_dict.word2idx(ans) for ans in valid_answers])
            ans_idx = (
                [self.answer_dict.word2idx(ans) for ans in valid_answers])
            answer_scores = (compute_answer_scores(ans_idx,
                                      self.answer_dict.num_vocab,
                                      self.answer_dict.UNK_idx))
        if answer is not None:
            answer_idx = self.answer_dict.word2idx(answer)

        if self.advanced_vqa:
            new_answer = self.tokenized_list[self.answer_dict.word2idx(answer)]
            subword_tokens = self.tokenizer.tokenize(" ".join(iminfo['question_tokens']))
            subword_tokens = ["[CLS]"] + subword_tokens + ["?"] # We will use the last word to do predictio

            masked_lm_labels = [-1] * len(subword_tokens)

            for i in new_answer:
                subword_tokens.append("[MASK]")
                masked_lm_labels.append(self.tokenizer.vocab[i])
            subword_tokens.append("[SEP]")
            masked_lm_labels.append(-1)

            input_ids = []
            for i in subword_tokens:
                input_ids.append(self.tokenizer.vocab[i])

            bert_feature = InputFeatures(
                unique_id = -1,
                tokens = subword_tokens,
                input_ids = input_ids,
                input_mask = [1] * len(input_ids),
                input_type_ids = [0] * len(input_ids),
                is_correct = 1, 
                lm_label_ids = masked_lm_labels
                )
            bert_feature.insert_field_into_dict(sample)
        else:
            if self.pretraining:
                item = iminfo
                if self.no_next_sentence:
                    answer = answer
                    label = None
                    subword_tokens_a = self.tokenizer.tokenize(" ".join(item['question_tokens'])) + ["?"] 
                    subword_tokens_b = self.tokenizer.tokenize(" ".join(answer))

                    bert_example = InputExample(unique_id = index, text_a = subword_tokens_a + subword_tokens_b, text_b = None, is_correct = None, max_seq_length = self.max_seq_length)
                    bert_feature = InputFeatures.convert_one_example_to_features_pretraining(
                                example = bert_example,
                                tokenizer=self.tokenizer,
                                probability = 0.15)
                else:
                    assert(0) # Should not use this part
                    '''if random.random() > self.false_caption_ratio:
                        answer = answer
                        label = 1
                    else:
                        while(True):
                            wrong_answer = np.random.choice(self.answer_dict.word_list)
                            if wrong_answer not in valid_answers:
                                wrong_answer = answer
                                label = 0
                                break
                    subword_tokens_a = self.tokenizer.tokenize(" ".join(item['question_tokens'])) + ["?"] 
                    subword_tokens_b = self.tokenizer.tokenize(" ".join(answer))
                    bert_example = InputExample(unique_id = index, text_a = subword_tokens_a, text_b = subword_tokens_b, is_correct = label, max_seq_length = self.max_seq_length)
                    bert_feature = InputFeatures.convert_one_example_to_features_pretraining(
                                example = bert_example,
                                tokenizer=self.tokenizer,
                                probability = 0.15)'''
                bert_feature.insert_field_into_dict(sample)
            else:
                item = iminfo
                subword_tokens = self.tokenizer.tokenize(" ".join(item['question_tokens']))
                if self.no_next_sentence:
                    subword_tokens = subword_tokens + ["?", "[MASK]"] # We will use the last word to do predictio
                    subwords_b = None
                else:
                    subword_tokens = subword_tokens + ["?"]
                    subwords_b = ["[MASK]"]
                bert_example = InputExample(unique_id = -1, text_a = subword_tokens, text_b = subwords_b,max_seq_length = self.max_seq_length)
                bert_feature = InputFeatures.convert_one_example_to_features(bert_example,tokenizer =self.tokenizer)
                bert_feature.insert_field_into_dict(sample)

        if answer is not None:
            sample['label'] = ArrayField(np.array(answer_scores))

        return Instance(sample)

    @staticmethod
    def collate_fn(data):
        if isinstance(data[0], Instance):
            batch = Batch(data)
            td = batch.as_tensor_dict()
            return td

    @classmethod
    def splits(cls, args):
        """ Helper method to generate splits of the dataset"""
        data_root = args.data_root

        if not args.use_visual_genome:
            assert(0) # This part should not be used
            '''chunk_train = torch.load(os.path.join(data_root, "coco/features_chunk_train.th"))
            chunk_val = torch.load(os.path.join(data_root, "coco/features_chunk_val.th"))
            print("Processing imges...")
            average = 0.0
            for image_id in chunk_train.keys():
                image_feat_variable, image_boxes, confidence = chunk_train[image_id]
                chunk_train[image_id] = screen_feature(image_feat_variable, image_boxes,confidence, args.image_screening_parameters)
                average += chunk_train[image_id][2]
            print("{} features on average.".format(average/len(chunk_train)))

            for image_id in chunk_val.keys():
                image_feat_variable, image_boxes, confidence = chunk_val[image_id]
                chunk_val[image_id] = screen_feature(image_feat_variable, image_boxes,confidence, args.image_screening_parameters)
                average += chunk_val[image_id][2]'''
        else:
            chunk_train = None
            chunk_val = None

        args_copy = deepcopy(args)
        args_copy.vocab_answer_file = os.path.join(data_root, "data/answers_vqa.txt")

        args_copy.imdb_file = [os.path.join(data_root, "data/imdb/imdb_train2014.npy"), os.path.join(data_root, "data/imdb/imdb_val2014.npy")] #imdb_val2train2014, imdb_val2014

        train = cls(args_copy, chunk_train = chunk_train, chunk_val = chunk_val)
        train.is_train = True

        args_copy_1 = deepcopy(args_copy)

        args_copy_1.imdb_file = os.path.join(data_root, "data/imdb/imdb_minival2014.npy")
        val = cls(args_copy_1, chunk_train = chunk_train, chunk_val = chunk_val)
        val.is_train = False

        args_copy_2 = deepcopy(args_copy)
        args_copy_2.imdb_file = os.path.join(data_root, "data/imdb/imdb_test2015.npy")
        test = cls(args_copy_2)
        test.is_train = False

        return train, val, test

    def generate_test_file(self, logits, out_file):
        assert(len(self.items) == logits.size(0))
        out_list = []
        for index, i in enumerate(self.items):
            question_id = i["question_id"]
            out_list.append(
            {
                "question_id": question_id,
                "answer": self.answer_dict.idx2word(logits[index].argmax(0))
            }
            )
        with open(out_file, "w") as f:
            json.dump(out_list, f)

import re
SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')

def tokenize(sentence):
    sentence = sentence.lower()
    sentence = (
        sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s'))
    tokens = SENTENCE_SPLIT_REGEX.split(sentence)
    tokens = [t.strip() for t in tokens if len(t.strip()) > 0]
    return tokens


def load_str_list(fname):
    with open(fname) as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    return lines


class VocabDict:
    def __init__(self, vocab_file):
        self.word_list = load_str_list(vocab_file)
        self.word2idx_dict = {w: n_w for n_w, w in enumerate(self.word_list)}
        self.num_vocab = len(self.word_list)
        self.UNK_idx = (self.word2idx_dict['<unk>']
                        if '<unk>' in self.word2idx_dict else None)

    def idx2word(self, n_w):
        return self.word_list[n_w]

    def word2idx(self, w):
        if w in self.word2idx_dict:
            return self.word2idx_dict[w]
        elif self.UNK_idx is not None:
            return self.UNK_idx
        else:
            raise ValueError('word %s not in dictionary \
                             (while dictionary does not contain <unk>)' % w)

    def tokenize_and_index(self, sentence):
        inds = [self.word2idx(w) for w in tokenize(sentence)]
        return inds
