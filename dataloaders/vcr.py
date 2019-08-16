# Modifed from R2C
"""
Dataloaders for VCR
"""
import json
import pickle
import os
from collections import defaultdict
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
from tqdm import tqdm

from .vcr_data_utils import data_iter, data_iter_test, data_iter_item

from .bert_data_utils import InputExample, InputFeatures, get_one_image_feature_npz_screening_parameters, get_image_feat_reader, faster_RCNN_feat_reader, screen_feature

from .bert_field import IntArrayField

from visualbert.pytorch_pretrained_bert.fine_tuning import _truncate_seq_pair, random_word
from visualbert.pytorch_pretrained_bert.tokenization import BertTokenizer

GENDER_NEUTRAL_NAMES = ['Casey', 'Riley', 'Jessie', 'Jackie', 'Avery', 'Jaime', 'Peyton', 'Kerry', 'Jody', 'Kendall',
                        'Peyton', 'Skyler', 'Frankie', 'Pat', 'Quinn']

# Here's an example jsonl
# {
# "movie": "3015_CHARLIE_ST_CLOUD",
# "objects": ["person", "person", "person", "car"],
# "interesting_scores": [0],
# "answer_likelihood": "possible",
# "img_fn": "lsmdc_3015_CHARLIE_ST_CLOUD/3015_CHARLIE_ST_CLOUD_00.23.57.935-00.24.00.783@0.jpg",
# "metadata_fn": "lsmdc_3015_CHARLIE_ST_CLOUD/3015_CHARLIE_ST_CLOUD_00.23.57.935-00.24.00.783@0.json",
# "answer_orig": "No she does not",
# "question_orig": "Does 3 feel comfortable?",
# "rationale_orig": "She is standing with her arms crossed and looks disturbed",
# "question": ["Does", [2], "feel", "comfortable", "?"],
# "answer_match_iter": [3, 0, 2, 1],
# "answer_sources": [3287, 0, 10184, 2260],
# "answer_choices": [
#     ["Yes", "because", "the", "person", "sitting", "next", "to", "her", "is", "smiling", "."],
#     ["No", "she", "does", "not", "."],
#     ["Yes", ",", "she", "is", "wearing", "something", "with", "thin", "straps", "."],
#     ["Yes", ",", "she", "is", "cold", "."]],
# "answer_label": 1,
# "rationale_choices": [
#     ["There", "is", "snow", "on", "the", "ground", ",", "and",
#         "she", "is", "wearing", "a", "coat", "and", "hate", "."],
#     ["She", "is", "standing", "with", "her", "arms", "crossed", "and", "looks", "disturbed", "."],
#     ["She", "is", "sitting", "very", "rigidly", "and", "tensely", "on", "the", "edge", "of", "the",
#         "bed", ".", "her", "posture", "is", "not", "relaxed", "and", "her", "face", "looks", "serious", "."],
#     [[2], "is", "laying", "in", "bed", "but", "not", "sleeping", ".",
#         "she", "looks", "sad", "and", "is", "curled", "into", "a", "ball", "."]],
# "rationale_sources": [1921, 0, 9750, 25743],
# "rationale_match_iter": [3, 0, 2, 1],
# "rationale_label": 1,
# "img_id": "train-0",
# "question_number": 0,
# "annot_id": "train-0",
# "match_fold": "train-0",
# "match_index": 0,
# }

class VCR(Dataset):
    def __init__(self, 
        split, 
        mode, 
        only_use_relevant_dets=True, 
        add_image_as_a_box=True, 
        conditioned_answer_choice=0, 
        do_lower_case = True, 
        bert_model_name = "", 
        max_seq_length = 128, 
        pretraining = False, 
        pretraining_include_qa_and_qar = False,
        complete_shuffle = False,
        use_alignment = False,

        add_all_features = False,
        answer_labels_path = None,
        vcr_annots_dir = None,
        vcr_image_dir = None
        ):
        # Should clean this mess when I find the time...

        self.split = split
        self.mode = mode
        self.only_use_relevant_dets = only_use_relevant_dets
        self.pretraining_include_qa_and_qar = pretraining_include_qa_and_qar

        self.add_all_features = add_all_features
        self.use_alignment = use_alignment

        self.add_image_as_a_box = add_image_as_a_box
        self.conditioned_answer_choice = conditioned_answer_choice

        self.vcr_annots_dir = vcr_annots_dir
        self.vcr_image_dir = vcr_image_dir
        with open(os.path.join(self.vcr_annots_dir, '{}.jsonl'.format(split)), 'r') as f:
            self.items = [json.loads(s) for s in f]

        if split not in ('test', 'train', 'val'):
            raise ValueError("Mode must be in test, train, or val. Supplied {}".format(mode))

        if mode not in ('answer', 'rationale'):
            raise ValueError("split must be answer or rationale")

        self.vocab = Vocabulary()

        with open(os.path.join(os.path.dirname(self.vcr_annots_dir), 'dataloaders', 'cocoontology.json'), 'r') as f:
            coco = json.load(f)
        self.coco_objects = ['__background__'] + [x['name'] for k, x in sorted(coco.items(), key=lambda x: int(x[0]))]
        self.coco_obj_to_ind = {o: i for i, o in enumerate(self.coco_objects)}


        self.do_lower_case = do_lower_case
        self.bert_model_name = bert_model_name

        self.max_seq_length = max_seq_length
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model_name, do_lower_case=self.do_lower_case)
        self.pretraining = pretraining

        # This is for pretraining
        self.masked_lm_prob = 0.15
        self.max_predictions_per_seq = 20

        self.complete_shuffle = complete_shuffle

        ##########
        self.only_qar = True if self.mode=='rationale' else False

        if answer_labels_path is not None:
            # Only when we are testing rationale...
            assert(self.only_qar)

            if answer_labels_path == 0:
                for index, i in enumerate(self.items):
                    i["answer_label"] = 0
            elif answer_labels_path == 1:
                for index, i in enumerate(self.items):
                    i["answer_label"] = 1
            elif answer_labels_path == 2:
                for index, i in enumerate(self.items):
                    i["answer_label"] = 2
            elif answer_labels_path == 3:
                for index, i in enumerate(self.items):
                    i["answer_label"] = 3
            else:
                self.answer_labels = np.load(answer_labels_path)
                self.answer_labels = self.answer_labels.argmax(1)
                if self.split == "test":
                    assert(self.answer_labels.shape[0] == len(self))
                    for index, i in enumerate(self.items):
                        i["answer_label"] = self.answer_labels[index]
        else:
            self.answer_labels = None

    @property
    def is_train(self):
        return self.split == 'train'

    @classmethod
    def splits(cls, **kwargs):
        """ Helper method to generate splits of the dataset"""
        kwargs_copy = {x: y for x, y in kwargs.items()}
        if 'mode' not in kwargs:
            kwargs_copy['mode'] = 'answer'

        train = cls(split='train', **kwargs_copy)
        val = cls(split='val', **kwargs_copy)
        test = cls(split='test', **kwargs_copy)
        return train, val, test

    def __len__(self):
        if self.complete_shuffle:
            if self.pretraining_include_qa_and_qar:
                return len(self.items) * 8
            else:
                return len(self.items) * 4
        return len(self.items)

    def _get_dets_to_use(self, item, only_use_answer = False, only_use_qar = False): # Need to fix this match
        """
        We might want to use fewer detectiosn so lets do so.
        :param item:
        :param question:
        :param answer_choices:
        :return:
        """
        # Load questions and answers
        question = item['question']
        answer_choices = item['{}_choices'.format(self.mode)]

        if self.mode == "answer":
            question = item['question']
            answer_choices = item['{}_choices'.format(self.mode)]
        elif self.mode == "rationale":
            question = item['question'] + item['answer_choices'][item['answer_label']]
            answer_choices = item['{}_choices'.format(self.mode)]

        if self.pretraining_include_qa_and_qar:
            answer_choices = item['answer_choices'] + item['rationale_choices']

        if self.add_all_features:
            question = item['question']
            answer_choices = item['answer_choices'] + item['rationale_choices']

        if self.only_use_relevant_dets:
            dets2use = np.zeros(len(item['objects']), dtype=bool)
            people = np.array([x == 'person' for x in item['objects']], dtype=bool)
            for sent in answer_choices + [question]:
                for possibly_det_list in sent:
                    if isinstance(possibly_det_list, list):
                        for tag in possibly_det_list:
                            if tag >= 0 and tag < len(item['objects']):  # sanity check
                                dets2use[tag] = True
                    elif possibly_det_list.lower() in ('everyone', 'everyones'):
                        dets2use |= people
            if not dets2use.any():
                dets2use |= people
        else:
            dets2use = np.ones(len(item['objects']), dtype=bool)

        # we will use these detections
        dets2use = np.where(dets2use)[0]

        old_det_to_new_ind = np.zeros(len(item['objects']), dtype=np.int32) - 1
        old_det_to_new_ind[dets2use] = np.arange(dets2use.shape[0], dtype=np.int32)

        # If we add the image as an extra box then the 0th will be the image.
        if self.add_image_as_a_box:
            old_det_to_new_ind[dets2use] += 1
        old_det_to_new_ind = old_det_to_new_ind.tolist()
        return dets2use, old_det_to_new_ind

    def __getitem__(self, index):

        if self.complete_shuffle:
            if self.pretraining_include_qa_and_qar:
                index = index // 8
                which = index % 8
            else:
                index = index // 4
                which = index % 4
        else:
            which = None

        item = deepcopy(self.items[index])

        ###################################################################
        # Load questions and answers
        
        answer_choices = item['{}_choices'.format(self.mode)]

        if self.complete_shuffle and which < 4:
            only_use_answer = True
        else:
            only_use_answer = False

        if self.complete_shuffle and which >= 4:
            only_use_qar = True
        else:
            only_use_qar = False

        dets2use, old_det_to_new_ind = self._get_dets_to_use(item, only_use_answer = only_use_answer, only_use_qar = only_use_qar)

        # The only_use_qar is ambigious...

        instance_dict = {}
        if self.split != 'test':
            instance_dict['label'] = LabelField(item['{}_label'.format(self.mode)], skip_indexing=True)
        instance_dict['metadata'] = MetadataField({'annot_id': item['annot_id'], 'ind': index, 'movie': item['movie'],
                                                   'img_fn': item['img_fn'],
                                                   'question_number': item['question_number']})

        ###################################################################
        # Load image now and rescale it. Might have to subtract the mean and whatnot here too.
        image = load_image(os.path.join(self.vcr_image_dir, item['img_fn']))
        #image = self.imagedatas(item['img_fn'])

        image, window, img_scale, padding = resize_image(image, random_pad=self.is_train)
        image = to_tensor_and_normalize(image)
        c, h, w = image.shape

        ###################################################################
        # Load boxes.
        with open(os.path.join(self.vcr_image_dir, item['metadata_fn']), 'r') as f:
            metadata = json.load(f)

        # [nobj, 14, 14]
        segms = np.stack([make_mask(mask_size=14, box=metadata['boxes'][i], polygons_list=metadata['segms'][i]) for i in dets2use])

        # Chop off the final dimension, that's the confidence
        boxes = np.array(metadata['boxes'])[dets2use, :-1]
        # Possibly rescale them if necessary
        boxes *= img_scale
        boxes[:, :2] += np.array(padding[:2])[None]
        boxes[:, 2:] += np.array(padding[:2])[None]
        obj_labels = [self.coco_obj_to_ind[item['objects'][i]] for i in dets2use.tolist()]
        if self.add_image_as_a_box:
            boxes = np.row_stack((window, boxes))
            segms = np.concatenate((np.ones((1, 14, 14), dtype=np.float32), segms), 0)
            obj_labels = [self.coco_obj_to_ind['__background__']] + obj_labels

        examples = data_iter_item(item, tokenizer=self.tokenizer,
                                            max_seq_length=self.max_seq_length,
                                            endingonly=False,
                                            include_qar = self.pretraining_include_qa_and_qar,
                                            only_qar = self.only_qar)
        self.getitem_bert_part(examples, item, instance_dict, which)

        if self.use_alignment: # Alignment between objects and text
            ######################
            examples_alginment_pack = []
            for i in range(len(examples)):
                if self.pretraining_include_qa_and_qar:
                    if i < 4:
                        raw_text_a = item["question"]
                        raw_text_b = item['answer_choices'][i]
                    else:
                        raw_text_a = item["question"] + item['answer_choices'][item['answer_label']]
                        raw_text_b = item['rationale_choices'][i - 4]
                elif self.only_qar:
                    raw_text_a = item["question"] + item['answer_choices'][item['answer_label']] # This is the correct alignment right now.
                    raw_text_b = item['rationale_choices'][i]
                else:
                    raw_text_a = item["question"]
                    raw_text_b = item['answer_choices'][i]

                true_text_a = examples[i][0].text_a
                true_text_b = examples[i][0].text_b
                text_alignment_a = examples[i][1]
                text_alignment_b = examples[i][2]

                examples_alginment_pack.append((raw_text_a, raw_text_b, true_text_a, true_text_b, text_alignment_a, text_alignment_b))

            image_box_position = []

            if which is not None:
                raw_text_a, raw_text_b, true_text_a, true_text_b, text_alignment_a, text_alignment_b = examples_alginment_pack[which]
                box_record = defaultdict(list)
                self.get_alignment_original(raw_text_a, text_alignment_a, old_det_to_new_ind, box_record, offset = 1)
                self.get_alignment_original(raw_text_b, text_alignment_b, old_det_to_new_ind, box_record, offset = 1 + len(text_alignment_a) + 1)
                image_text_alignment = ListField([IntArrayField(np.array(box_record[i]), padding_value = -1) for i in range(len(boxes))])
            else:
                for raw_text_a, raw_text_b, true_text_a, true_text_b, text_alignment_a, text_alignment_b in examples_alginment_pack:

                    box_record = defaultdict(list)
                    self.get_alignment_original(raw_text_a, text_alignment_a, old_det_to_new_ind, box_record, offset = 1)
                    self.get_alignment_original(raw_text_b, text_alignment_b, old_det_to_new_ind, box_record, offset = 1 + len(text_alignment_a) + 1)

                    image_box_position.append(ListField([IntArrayField(np.array(box_record[i]), padding_value = -1) for i in range(len(boxes))]))

                image_text_alignment = ListField(image_box_position)
            ######################

            instance_dict["image_text_alignment"] = image_text_alignment

        instance_dict['segms'] = ArrayField(segms, padding_value=0)
        instance_dict['objects'] = ListField([LabelField(x, skip_indexing=True) for x in obj_labels])

        if not np.all((boxes[:, 0] >= 0.) & (boxes[:, 0] < boxes[:, 2])):
            import ipdb
            ipdb.set_trace()
        assert np.all((boxes[:, 1] >= 0.) & (boxes[:, 1] < boxes[:, 3]))
        assert np.all((boxes[:, 2] <= w))
        assert np.all((boxes[:, 3] <= h))
        instance_dict['boxes'] = ArrayField(boxes, padding_value=-1)

        instance = Instance(instance_dict)
        instance.index_fields(self.vocab)
        return image, instance

    def get_alignment_original(self, raw_text_mixed, text_alignment, old_det_to_new_ind, box_record, offset):
        # raw_text_mixed is the raw text information in VCR dataset
        # text_alignment is the result from BERT tokenizer recording the alignment between raw tokens and subword tokens.
        counter = 0
        for i in raw_text_mixed:
            if isinstance(i, list):
                for box_index in i:
                    new_box_index = old_det_to_new_ind[box_index]
                    assert(new_box_index != -1)
                    # Need to record which box corresponds to which person.
                    for i in text_alignment:
                        if i == counter:
                            box_record[new_box_index].append(i + offset)
                            break
                    counter += 1
            else:
                counter += 1

    def getitem_bert_part(self, examples, item, instance_dict, which = None):
        # In examples, each element: InputExample, Alignment for context, algiment for answer
        if self.pretraining:
            if self.complete_shuffle:
                assert(which is not None)
                feature = InputFeatures.convert_one_example_to_features_pretraining(
                        example = examples[which][0],
                        tokenizer=self.tokenizer,
                        probability = self.masked_lm_prob)
                feature.insert_field_into_dict(instance_dict)
                return

            features = []
            for i in examples:
                inputexample_instance = i[0]
                example = InputFeatures.convert_one_example_to_features_pretraining(
                        example = inputexample_instance,
                        tokenizer=self.tokenizer,
                        probability = self.masked_lm_prob)
                features.append(example)
            InputFeatures.convert_list_features_to_allennlp_list_feild(features, instance_dict)

        else:
            features = InputFeatures.convert_examples_to_features(
                    examples=[x[0] for x in examples],
                    tokenizer=self.tokenizer)
            InputFeatures.convert_list_features_to_allennlp_list_feild(features, instance_dict) 

    @staticmethod
    def collate_fn(data):
        if isinstance(data[0], Instance):
            batch = Batch(data)
            td = batch.as_tensor_dict()
            return td
        else:
            images, instances = zip(*data)
            images = torch.stack(images, 0)

            batch = Batch(instances)
            td = batch.as_tensor_dict()
            if 'question' in td:
                td['question_mask'] = get_text_field_mask(td['question'], num_wrapping_dims=1)
                td['question_tags'][td['question_mask'] == 0] = -2  # Padding
            if "answer" in td:
                td['answer_mask'] = get_text_field_mask(td['answers'], num_wrapping_dims=1)
                td['answer_tags'][td['answer_mask'] == 0] = -2

            td['box_mask'] = torch.all(td['boxes'] >= 0, -1).long()
            td['images'] = images
            return td


class VCRLoader(torch.utils.data.DataLoader):
    """
    Iterates through the data, filtering out None,
     but also loads everything as a (cuda) variable
    """

    @classmethod
    def from_dataset(cls, data, batch_size=3, num_workers=6, num_gpus=3, **kwargs):
        loader = cls(
            dataset=data,
            batch_size=batch_size * num_gpus,
            shuffle=data.is_train,
            num_workers=num_workers,
            collate_fn=data.collate_fn,
            drop_last=False,
            pin_memory=False,
            **kwargs,
        )
        return loader