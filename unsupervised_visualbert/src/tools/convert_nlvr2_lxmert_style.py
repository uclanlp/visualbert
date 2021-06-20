#{'img_id': 'COCO_train2014_000000318556', 'labelf': {'vqa': [{'no': 1}, {'yes': 1}, {'no': 1}, {'blue': 1, 'blue and white': 0.3}]}, 'sentf': {'mscoco': ['A very clean and well decorated empty bathroom', 'A blue and white bathroom with butterfly themed wall tiles.', 'A bathroom with a border of butterflies and blue paint on the walls above it.', 'An angled view of a beautifully decorated bathroom.', 'A clock that blends in with the wall hangs in a bathroom. '], 'vqa': ['Is the sink full of water?', 'Are there any butterflies on the tiles?', 'Is this bathroom in a hotel?', 'What color are the walls?']}}
import json
split = "valid"

target = "/local/harold/ubert/lxmert/data/lxmert/nlvr_for_pretrain_{}.json".format(split)

train_file_name = "/local/harold/ubert/lxmert/data/nlvr2/{}.json".format(split)
train_data = []

with open(train_file_name) as f:
  data = json.load(f)


'''
{'identifier': 'train-10171-0-0', 'img0': 'train-10171-0-img0', 'img1': 'train-10171-0-img1', 'label': 0, 'sent': 'An image shows one leather pencil case, displayed open with writing implements tucked inside.', 'uid': 'nlvr2_train_0'}
'''

for one_data in data:
    

    one_datatum = {}
    one_datatum["img_id"] = one_data["img0"]
    one_datatum["img_id_1"] = one_data["img1"]

    one_datatum["uid"] = one_data["uid"]
    one_datatum["identifier"] = one_data["identifier"]
    one_datatum["label"] =  one_data["label"]
    
    one_datatum["labelf"] = {}
    one_datatum["sentf"] = {}
    one_datatum["sentf"]["nlvr"] = []
    one_datatum["sentf"]["nlvr"].append(one_data["sent"])
    train_data.append(one_datatum)

with open(target, 'w') as f:
    json.dump(train_data, f)