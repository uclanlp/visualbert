#{'img_id': 'COCO_train2014_000000318556', 'labelf': {'vqa': [{'no': 1}, {'yes': 1}, {'no': 1}, {'blue': 1, 'blue and white': 0.3}]}, 'sentf': {'mscoco': ['A very clean and well decorated empty bathroom', 'A blue and white bathroom with butterfly themed wall tiles.', 'A bathroom with a border of butterflies and blue paint on the walls above it.', 'An angled view of a beautifully decorated bathroom.', 'A clock that blends in with the wall hangs in a bathroom. '], 'vqa': ['Is the sink full of water?', 'Are there any butterflies on the tiles?', 'Is this bathroom in a hotel?', 'What color are the walls?']}}
import json

target = "/local/harold/vqa/google_concetual/train"
with open("%s.tsv" % target, 'r') as f:
  lines = f.readlines()


train_file_name = "/local/harold/ubert/lxmert/data/lxmert/{}.json".format("google_cc_train" if "train" in target else "google_cc_valid")
train_data = []

for i in range(len(lines)):
    caption, url = lines[i].strip('\n').split("\t", 1)

    one_datatum = {}
    one_datatum["img_id"] = "{}/{}.jpg".format(target, i)
    one_datatum["labelf"] = {}
    one_datatum["sentf"] = {}
    one_datatum["sentf"]["google_cc"] = []
    one_datatum["sentf"]["google_cc"].append(caption)
    train_data.append(one_datatum)

with open(train_file_name, 'w') as f:
    json.dump(train_data, f)