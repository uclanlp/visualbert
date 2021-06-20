import random
from torch.utils.data import Dataset
from lxrt.tokenization import BertTokenizer
import logging
from lxmert_data import InputExample
import json
from param import args
from lxmert_data import InputFeatures, random_word
import os
from src.tools import sharearray
import gc
from tqdm import tqdm
import numpy as np

class GeneralCorpusNP(Dataset):
    def __init__(self, ann_file, pretrained_model_name, tokenizer=None, seq_len=64, min_seq_len=64,
                 encoding="utf-8", on_memory=True,
                 **kwargs):
        assert on_memory, "only support on_memory mode!"

        self.tokenizer = tokenizer if tokenizer is not None else BertTokenizer.from_pretrained(pretrained_model_name)
        self.vocab = self.tokenizer.vocab
        self.seq_len = seq_len
        self.min_seq_len = min_seq_len
        self.on_memory = on_memory
        self.ann_file = ann_file
        self.encoding = encoding
        self.test_mode = False
        
        self.do_no_fill = False

        self.use_mismatch_objective = args.get("task_matched", False)
        #self.load_corpus_with_passages()

        # load samples into memory
        if on_memory:
            if self.use_mismatch_objective:
                #self.corpus = self.load_corpus_with_passages_preprocess()
                self.load_corpus_with_passages_preprocess()
            else:
                self.corpus = self.load_corpus()
        if args.get("presegment_sentence", False):
            self.presegment_sentence()
        print("Using {} with {} data.\n\n".format(self.ann_file, len(self)))

    def load_corpus(self):
        corpus = []

        print("\n\nLoading text only corpus...")
        for ann_file in self.ann_file.split('+'):
            with open(ann_file, 'r', encoding=self.encoding) as f:
                all_text = f.read().lower()
                corpus.extend([l.strip('\n').strip('\r').strip('\n') for l in all_text.split("\n")])

        corpus = [l.strip() for l in corpus if l.strip() != '']
        return corpus
    
    def load_corpus_with_passages_preprocess(self):
        corpus = []

        print("\n\nLoading text only corpus...")

        if os.path.exists(args.text_only_corpus_cache):
            with open(args.text_only_corpus_cache, 'rb') as f:
                corpus = np.load(f)
                self.corpus = sharearray.cache(self.ann_file.split("/")[-1], corpus)
                del corpus
                gc.collect()

            with open(args.text_only_corpus_cache.replace("npy", "json"), 'r') as f:
                files = json.load(f)
            [self.passage_split, self.sentence_split] = files

            self.sentence_counter = [0] * len(self.sentence_split) 
        else:
            new_text = []
            passage_split = []
            sentence_split = []
            current_counter = 0

            for ann_file in self.ann_file.split('+'):
                with open(ann_file, 'r', encoding=self.encoding) as f:
                    all_text = f.read().lower()
                    one_passage_sentence_split = []
                    counter = 0

                    for line in tqdm(all_text.split("\n")):
                        line = line.strip('\n').strip('\r').strip('\n')

                        line = self.tokenizer.wordpiece_tokenizer.tokenize(line)
                        line_ids = self.tokenizer.convert_tokens_to_ids(line)

                        if len(line) != 0:
                            new_text.extend(line_ids)
                            counter += len(line_ids)
                            one_passage_sentence_split.append(counter)
                        else:
                            if counter != 0:
                                #all_text.extend(one_passage)
                                sentence_split.append(one_passage_sentence_split)
                                current_counter += counter
                                passage_split.append(current_counter)

                            one_passage = []
                            one_passage_sentence_split = []
                            counter = 0
            
                    #corpus.extend([l.strip('\n').strip('\r').strip('\n') for l in all_text.split("\n")])

            #corpus = [l.strip() for l in corpus if l.strip() != '']
        
            self.sentence_counter = [0] * len(passage_split)  # we keep a record of when 
            self.corpus = np.array(new_text)
            
            self.passage_split = passage_split
            self.sentence_split = sentence_split
            with open(args.text_only_corpus_cache, 'wb') as f:
                np.save(f, self.corpus)

            with open(args.text_only_corpus_cache.replace("npy", "json"), 'w') as f:
                json.dump([self.passage_split, self.sentence_split], f)
            assert(0)

    #def save_sentence_counter(self):
    #

    def __len__(self):
        if args.get("presegment_sentence", False) and "sbu-captions-all.json" not in self.ann_file:
            return len(self.mapping)
        return len(self.passage_split)

    def retrieve_a_piece(self, index, seq_len):
        if index == 0:
            begin = 0
        else:
            begin = self.passage_split[index - 1]
        
        end = self.passage_split[index]

        text = self.corpus[begin:end]
        sentence_split = self.sentence_split[index]

        ## Retrive part of 
        start_index = self.sentence_counter[index]

        all_tokenized_words = []
        all_mlm_labels = []
        current_length = 0
        final_index = -1
        
        for i in range(start_index, len(sentence_split)):
            if i == 0:
                begin = 0
            else:
                begin = sentence_split[i - 1]
            end = sentence_split[i]
            tokens = self.tokenizer.convert_ids_to_tokens(text[begin:end])
            tokens, mlm_labels = self.random_word_wwm(tokens)
            if current_length == 0 or len(tokens) + current_length <= seq_len:
                all_tokenized_words.extend(tokens)
                all_mlm_labels.extend(mlm_labels)
                current_length += len(tokens)
                final_index = (i + 1) % len(sentence_split)
            else:
                final_index = (i + 1) % len(sentence_split)
                break
        self.sentence_counter[index] = final_index  # Start from here next time retrieve a piece is called; Not sure how this will behave if we have multiple workers...
        #print(index, self.sentence_counter[index])
        all_tokenized_words = all_tokenized_words[:seq_len]
        all_mlm_labels = all_mlm_labels[:seq_len]
        
        return all_tokenized_words, all_mlm_labels
    
    def exhaustively_retrieve_a_piece(self, index, seq_len):
        all_ranges = []

        if index == 0:
            begin = 0
        else:
            begin = self.passage_split[index - 1]
        
        end = self.passage_split[index]

        text = self.corpus[begin:end]
        sentence_split = self.sentence_split[index]

        ## Retrive part of 
        start_index = 0 #self.sentence_counter[index]
        while True:
        
            all_tokenized_words = []
            all_mlm_labels = []
            current_length = 0
            final_index = -1

            sent_begin = 0
            sent_end = 0
            for i in range(start_index, len(sentence_split)):
                if i == 0:
                    sent_begin = 0
                else:
                    sent_begin = sentence_split[i - 1]
                tmp_sent_end = sentence_split[i]

                if current_length == 0 or (tmp_sent_end - sent_begin) + current_length <= seq_len:
                    current_length += tmp_sent_end - sent_begin
                    sent_end = tmp_sent_end
                    final_index = (i + 1) % len(sentence_split)
                else:
                    final_index = (i + 1) % len(sentence_split)
                    break

            if start_index == 0:
                sent_begin = 0
            else:
                sent_begin = sentence_split[start_index - 1]

            start_index = final_index

            all_ranges.append((begin + sent_begin, begin + sent_end))

            if start_index == 0:
                break

        return all_ranges
    
    def presegment_sentence(self):
        all_segments = []
        self.mapping = {}
        
        current_len = 0
        for i in tqdm(range(len(self.passage_split))):
            tmp = self.exhaustively_retrieve_a_piece(i, self.seq_len // 2)
            for j in range(len(tmp)):
                self.mapping[current_len + j] = current_len + (j + 1)%len(tmp)
            current_len += len(tmp)
            all_segments.extend(tmp)        
        self.all_segments = all_segments

    def retrieve_a_piece_preseged(self, index, seq_len):
        seg = self.all_segments[index]
        tokens = self.tokenizer.convert_ids_to_tokens(self.corpus[seg[0]:seg[1]])
        tokens, mlm_labels = self.random_word_wwm(tokens)
        tokens = tokens[:seq_len]
        mlm_labels = mlm_labels[:seq_len]
        return tokens, mlm_labels

        
    def __getitem__(self, item):
        if self.use_mismatch_objective:
            i = 0
            max_seq_length = self.seq_len // 2 # We have two parts
            
            if args.get("presegment_sentence", False) and "sbu-captions-all.json" not in self.ann_file:
                text_a_tokens, text_a_labels = self.retrieve_a_piece_preseged(item, seq_len = max_seq_length)

                # First we take out some sentences
                if random.random() < 0.5:
                    # Take out our own 
                    b_index = self.mapping[item]
                    text_b_tokens, text_b_labels = self.retrieve_a_piece_preseged(b_index, seq_len=max_seq_length)
                    match = 1
                else:
                    random_index = random.randint(0, len(self) - 1)
                    while random_index == item:
                        random_index = random.randint(0, len(self) - 1)
                    text_b_tokens, text_b_labels = self.retrieve_a_piece_preseged(random_index, seq_len=max_seq_length)
                    match = 0

            else:
                text_a_tokens, text_a_labels = self.retrieve_a_piece(item, seq_len = max_seq_length)

                # First we take out some sentences
                if random.random() < 0.5:
                    # Take out our own 
                    text_b_tokens, text_b_labels = self.retrieve_a_piece(item, seq_len=max_seq_length)
                    match = 1
                else:
                    random_index = random.randint(0, len(self) - 1)
                    while random_index == item:
                        random_index = random.randint(0, len(self) - 1)
                    text_b_tokens, text_b_labels = self.retrieve_a_piece(random_index, seq_len=max_seq_length)
                    match = 0

            text_a_ids = self.tokenizer.convert_tokens_to_ids(text_a_tokens)
            text_b_ids = self.tokenizer.convert_tokens_to_ids(text_b_tokens)

            example = InputExample(
                None, (text_a_tokens, text_b_tokens), (None, None),
                (None, None), (None, None),
                match, 1,
                mlm_labels=(text_a_labels, text_b_labels),
                token_ids=(text_a_ids, text_b_ids),
                max_seq_len = self.seq_len + 3
            )
            if args.get("faster_loading", False):
                return self.convert_example_to_features(example, self.seq_len + 3, self.tokenizer)

        raw = self.corpus[item]

        # tokenize
        tokens = self.tokenizer.basic_tokenizer.tokenize(raw.lower())

        if not self.do_no_fill:
            # add more tokens if len(tokens) < min_len
            _cur = (item + 1) % len(self.corpus)
            while len(tokens) < self.min_seq_len:
                _cur_tokens = self.tokenizer.basic_tokenizer.tokenize(self.corpus[_cur])
                tokens.extend(_cur_tokens)
                _cur = (_cur + 1) % len(self.corpus)

        # masked language modeling
        tokens, mlm_labels = self.random_word_wwm(tokens)

        # convert token to its vocab id
        ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # truncate
        if len(ids) > self.seq_len:
            ids = ids[:self.seq_len]
            mlm_labels = mlm_labels[:self.seq_len]
        
        example = InputExample(
            None, tokens, (None, None),
            (None, None), (None, None),
            None, 1,
            mlm_labels=mlm_labels,
            token_ids=ids,
            max_seq_len = self.seq_len
        )
        if args.get("faster_loading", False):
            return self.convert_example_to_features(example, args.get("max_seq_length", 20), self.tokenizer)

        return example

    def convert_example_to_features(self, example: InputExample, max_seq_length, tokenizer, hybrid_num=10):
        if isinstance(example.mlm_labels, tuple):
            text_a_ids, text_b_ids = example.token_ids
            text_a_labels, text_b_labels = example.mlm_labels
            input_ids = tokenizer.convert_tokens_to_ids(["[CLS]"]) + text_a_ids + tokenizer.convert_tokens_to_ids(["[SEP]"]) + text_b_ids + tokenizer.convert_tokens_to_ids(["[SEP]"])
            lm_label_ids = [-1] + text_a_labels + [-1] + text_b_labels + [-1]
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            
            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                lm_label_ids.append(-1)

            features = InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                lm_label_ids=lm_label_ids,
                visual_feats=(None, None),
                obj_labels={
                    'obj': (None, None),
                    'attr': (None, None),
                    'feat': (None, None),
                },
                is_matched=example.is_matched,
                ans=-1,
                visual_tags = None,
                visual_tags_objective = None,
                visual_tags_mask = None,
                visual_tags_box=None,
                visual_tags_mismatch=None
            )
            return features


        if example.mlm_labels is not None:  # The data is already pre-masked
            input_ids = example.token_ids
            lm_label_ids = example.mlm_labels
            max_seq_len = example.max_seq_len + 2
            # Add [CLS] and [SEP]
            input_ids = tokenizer.convert_tokens_to_ids(["[CLS]"]) + input_ids + tokenizer.convert_tokens_to_ids(["[SEP]"])
            lm_label_ids = [-1] + lm_label_ids + [-1]
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_len:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                lm_label_ids.append(-1)

            features = InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                lm_label_ids=lm_label_ids,
                visual_feats=(None, None),
                obj_labels={
                    'obj': (None, None),
                    'attr': (None, None),
                    'feat': (None, None),
                },
                is_matched=1,
                ans=-1,
                visual_tags = None,
                visual_tags_objective = None,
                visual_tags_mask = None,
                visual_tags_box=None,
                visual_tags_mismatch=None
            )

            return features



    def random_word_wwm(self, tokens):
        output_tokens = []
        output_label = []

        for i, token in enumerate(tokens):
            sub_tokens = self.tokenizer.wordpiece_tokenizer.tokenize(token)
            prob = random.random()
            # mask token with 15% probability
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    for sub_token in sub_tokens:
                        output_tokens.append("[MASK]")
                # 10% randomly change token to random token
                elif prob < 0.9:
                    for sub_token in sub_tokens:
                        output_tokens.append(random.choice(list(self.tokenizer.vocab.keys())))
                        # -> rest 10% randomly keep current token
                else:
                    for sub_token in sub_tokens:
                        output_tokens.append(sub_token)

                        # append current token to output (we will predict these later)
                for sub_token in sub_tokens:
                    try:
                        output_label.append(self.tokenizer.vocab[sub_token])
                    except KeyError:
                        # For unknown words (should not occur with BPE vocab)
                        output_label.append(self.tokenizer.vocab["[UNK]"])
                        logging.warning("Cannot find sub_token '{}' in vocab. Using [UNK] insetad".format(sub_token))
            else:
                for sub_token in sub_tokens:
                    # no masking token (will be ignored by loss function later)
                    output_tokens.append(sub_token)
                    output_label.append(-1)

        ## if no word masked, random choose a word to mask
        # if all([l_ == -1 for l_ in output_label]):
        #    choosed = random.randrange(0, len(output_label))
        #    output_label[choosed] = self.tokenizer.vocab[tokens[choosed]]

        return output_tokens, output_label


def string_to_sequence(s: str, dtype=np.int32) -> np.ndarray:
    return np.array([ord(c) for c in s], dtype=dtype)

def sequence_to_string(seq: np.ndarray) -> str:
    return ''.join([chr(c) for c in seq])