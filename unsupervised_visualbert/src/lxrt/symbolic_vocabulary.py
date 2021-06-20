from param import args

class SymbolicVocab:
    def __init__(self, object_path, attribute_path, cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]", take_fisrt = True):
        attributes = []
        with open(attribute_path) as f:
            for line in f:
                attr = line.strip("\n")
                if "," in attr and take_fisrt:
                    attr = attr.split(",")[0]

                if len(attr) != 0:
                    attributes.append(attr)
        assert (len(attributes) == 400)

        objects = []
        with open(object_path) as f:
            for line in f:
                attr = line.strip("\n")
                if "," in attr and take_fisrt:
                    attr = attr.split(",")[0]
                if len(attr) != 0:
                    objects.append(attr)
        assert (len(objects) == 1600)

        self.attributes = attributes
        self.objects = objects

        self.id2word = []
        self.id2word.append(cls_token)
        self.id2word.append(sep_token)
        self.id2word.append(mask_token)
        self.id2word.extend(attributes)
        self.id2word.extend(objects)

        self.length_of_attribute = len(attributes)

        self.word2id = {}
        for index, word in enumerate(self.id2word):
            self.word2id[word] = index
    
    def __len__(self):
        return self.id2word

    def obj_id2word(self, index):
        return self.objects[index]
    
    def attr_id2word(self, index):
        return self.attributes[index]

    def get_symbolic_list(self, tokenizer):
        all_subwords = []
        for word in self.id2word:
            all_subwords.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word)))
        return all_subwords
    
    def get_seg_id(self, word_id):
        if word_id >= 3 and word_id < self.length_of_attribute + 3:
            return 1
        return 0
    