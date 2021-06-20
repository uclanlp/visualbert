import numpy as np
import torch.nn as nn

from param import args
from lxrt.entry import LXRTEncoder
from lxrt.modeling import BertLayerNorm, GeLU
from lxrt.tokenization import BertTokenizer
import torch
import numpy as np
from collections import defaultdict
import numpy
import random
'''
Given that tags will be extensively used now, writing some snippets for creating tags.
'''

def pad_np_arrays(list_of_np_array, padding_value, dtype):
    if isinstance(list_of_np_array[0], list):
        list_of_np_array = [np.array(i, dtype = dtype) for i in list_of_np_array]

    padding_lengths = get_padding_lengths(list_of_np_array)

    max_shape = [padding_lengths["dimension_{}".format(i)]
                    for i in range(len(padding_lengths))]

    # Convert explicitly to an ndarray just in case it's an scalar (it'd end up not being an ndarray otherwise)
    final_list = []

    for array_index, array in enumerate(list_of_np_array):
        return_array = numpy.asarray(numpy.ones(max_shape, dtype = dtype) * padding_value)
        # If the tensor has a different shape from the largest tensor, pad dimensions with zeros to
        # form the right shaped list of slices for insertion into the final tensor.
        slicing_shape = list(array.shape)
        #if len(array.shape) < len(max_shape):
        #    slicing_shape = slicing_shape + [0 for _ in range(len(max_shape) - len(array.shape))]
        slices = tuple([slice(0, x) for x in slicing_shape])

        return_array[slices] = array
        final_list.append(return_array)
    final_list = np.stack(final_list, 0)
    tensor = torch.from_numpy(final_list)
    return tensor

def transfer_object_labels_to_symbolic_ids(obj_labels, attribute_labels, symbolic_vocab, obj_confs = None, attr_confs = None):
    return_list = []

    for index in range(len(obj_labels)):
        prob = random.random()
        if prob < args.get("insert_attr_ratio", 0.0):
            if args.get("kl_divergence", False):
                if args.get("non_top1_sampling", False):
                    p = attr_confs[index][attribute_labels[index]]
                    p = p / p.sum()
                    attr_label_i = np.random.choice(attribute_labels[index], p=p)
                    #attr_label_i = np.random.choice(attr_confs.shape[-1], p=attr_confs[index])
                else:
                    attr_label_i = attribute_labels[index, 0]
            else:
                attr_label_i = attribute_labels[index]
            return_list.append(symbolic_vocab.word2id[symbolic_vocab.attr_id2word(attr_label_i)])
        else:
            if args.get("kl_divergence", False):
                if args.get("non_top1_sampling", False):
                    new_obj_confs = deepcopy(obj_confs)
                    new_obj_confs[new_obj_confs<0.1] = 0
                    p = new_obj_confs[index][obj_labels[index]]
                    sum_p = p.sum()
                    if sum_p == 0:
                        obj_label_i = obj_labels[index, 0]
                    else:
                        p = p / sum_p
                        obj_label_i =np.random.choice(obj_labels[index], p=p)
                        #obj_label_i = np.random.choice(obj_confs.shape[-1], p=obj_confs[index])
                else:
                    obj_label_i = obj_labels[index, 0]
            else:
                obj_label_i = obj_labels[index]
            return_list.append(symbolic_vocab.word2id[symbolic_vocab.obj_id2word(obj_label_i)])
    return np.array(return_list, dtype=np.int64)

def convert_semantic_objective(labels, symbolic_vocab, obj = False, attr = False, tokenizer=None):
    if obj:
        words = [symbolic_vocab.obj_id2word(i) for i in labels]
    elif attr:
        words = [symbolic_vocab.attr_id2word(i) for i in labels]
    else:
        assert(0)
    words = [symbolic_vocab.id2objective[symbolic_vocab.word2id[i]] for i in words]
    semantic_objective = np.array(words, dtype=np.int64) # object_num * 2
    return semantic_objective
    
def create_tags_pretrain(obj_labels, attr_labels, obj_confs, attr_confs, tokenizer, symbolic_vocab, visual_tags_box, feat_mask, use_bert_input = True):
    obj_labels_transformed = transfer_object_labels_to_symbolic_ids(obj_labels, attr_labels, symbolic_vocab, obj_confs, attr_confs)
    visual_tags_bert_words = []
    visual_tags_box_bert_input = []
    visual_tags_mlm_labels = []
    visual_tags_segment_ids = []

    for tag_index, tag in enumerate(obj_labels_transformed):
        tag_word = symbolic_vocab.id2word[tag]
        if args.get("use_segment_id_for_attr", False):
            seg_id = symbolic_vocab.get_seg_id(tag)
        sub_tokens = tokenizer.tokenize(tag_word)

        prob = random.random() 
        if prob < args.get('tag_mask_ratio', 0.15) or (feat_mask[tag_index] != 0 and random.random() < args.get("tag_joint_mask_ratio", 0.5)):

            new_prob = random.random()
            if new_prob < 0.8:
                for sub_token in sub_tokens:
                    visual_tags_bert_words.append("[MASK]")
            elif new_prob < 0.9:
                for sub_token in sub_tokens:
                    visual_tags_bert_words.append(random.choice(list(tokenizer.vocab.keys())))
            else:
                visual_tags_bert_words.extend(sub_tokens)
            
            for sub_token in sub_tokens:
                try:
                    visual_tags_mlm_labels.append(tokenizer.vocab[sub_token])
                except KeyError:
                    # For unknown words (should not occur with BPE vocab)
                    visual_tags_mlm_labels.append(tokenizer.vocab["[UNK]"])
                    logging.warning("Cannot find sub_token '{}' in vocab. Using [UNK] insetad".format(sub_token))

        else:
            for sub_token in sub_tokens:
                # no masking token (will be ignored by loss function later)
                visual_tags_bert_words.append(sub_token)
                visual_tags_mlm_labels.append(-1)

        # duplicate box
        for sub_token in sub_tokens:
            visual_tags_box_bert_input.append(visual_tags_box[tag_index])
            if args.get("use_segment_id_for_attr", False):
                visual_tags_segment_ids.append(seg_id)
    visual_tags = tokenizer.convert_tokens_to_ids(visual_tags_bert_words)
    visual_tags_objective = visual_tags_mlm_labels
    visual_tags_mask = [1] * len(visual_tags)
    visual_tags_box = visual_tags_box_bert_input

    visual_tags_segment_ids = None

    return visual_tags, visual_tags_objective, visual_tags_mask, visual_tags_box, visual_tags_segment_ids

def create_tags(obj_labels, attr_labels, obj_confs, attr_confs, tokenizer, symbolic_vocab, visual_tags_box, use_bert_input = True, record_index = None):
    obj_labels_transformed = transfer_object_labels_to_symbolic_ids(obj_labels, attr_labels, symbolic_vocab, obj_confs, attr_confs)
    visual_tags_bert_words = []
    visual_tags_box_bert_input = []
    #visual_tags_mlm_labels = []
    visual_tags_segment_ids = []
    
    recorded_indexes = []
    counter = 0
    for tag_index, tag in enumerate(obj_labels_transformed):
        tag_word = symbolic_vocab.id2word[tag]
        if args.get("use_segment_id_for_attr", False):
            seg_id = symbolic_vocab.get_seg_id(tag)
        sub_tokens = tokenizer.tokenize(tag_word)

        for sub_token in sub_tokens:
            # no masking token (will be ignored by loss function later)
            visual_tags_bert_words.append(sub_token)
            #visual_tags_mlm_labels.append(-1)
            if tag_index == record_index:
                recorded_indexes.append(counter)

            counter += 1

        # duplicate box
        for sub_token in sub_tokens:
            visual_tags_box_bert_input.append(visual_tags_box[tag_index])
            if args.get("use_segment_id_for_attr", False):
                visual_tags_segment_ids.append(seg_id)

    visual_tags = tokenizer.convert_tokens_to_ids(visual_tags_bert_words)
    visual_tags_mask = [1] * len(visual_tags)
    visual_tags_box = visual_tags_box_bert_input
    visual_tags_segment_ids = None
    visual_tags_type = None

    if record_index is not None:
        return visual_tags, visual_tags_mask, visual_tags_box, visual_tags_type, visual_tags_segment_ids, recorded_indexes

    return visual_tags, visual_tags_mask, visual_tags_box, visual_tags_type, visual_tags_segment_ids