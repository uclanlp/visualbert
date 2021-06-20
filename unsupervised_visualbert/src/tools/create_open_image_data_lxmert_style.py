#{'img_id': 'COCO_train2014_000000318556', 'labelf': {'vqa': [{'no': 1}, {'yes': 1}, {'no': 1}, {'blue': 1, 'blue and white': 0.3}]}, 'sentf': {'mscoco': ['A very clean and well decorated empty bathroom', 'A blue and white bathroom with butterfly themed wall tiles.', 'A bathroom with a border of butterflies and blue paint on the walls above it.', 'An angled view of a beautifully decorated bathroom.', 'A clock that blends in with the wall hangs in a bathroom. '], 'vqa': ['Is the sink full of water?', 'Are there any butterflies on the tiles?', 'Is this bathroom in a hotel?', 'What color are the walls?']}}
import json
import os
target = "/local/harold/ubert/lxmert/data/open_image/butd_feat/"

all_image_files = []

for root, dirs, files in os.walk(target, topdown=False):
    for txt_file in files:
        if txt_file.endswith("txt"):
            with open(os.path.join(target, txt_file)) as f:
              lines = f.read().split("\n")
              for line in lines:
                if len(line) != 0:
                  all_image_files.append(line)



train_file_name = "/local/harold/ubert/lxmert/data/lxmert/open_images_train.json" #.format("open_images_train" if "train" in target else "open_images_valid")
train_data = []

for i in range(len(all_image_files)):
    #caption, url = lines[i].strip('\n').split("\t", 1)

    one_datatum = {}
    one_datatum["img_id"] = all_image_files[i] #"{}/{}.jpg".format(target, i)
    one_datatum["labelf"] = {}
    one_datatum["sentf"] = {}
    one_datatum["sentf"]["open_image"] = []
    one_datatum["sentf"]["open_image"].append("")
    train_data.append(one_datatum)

with open(train_file_name, 'w') as f:
    json.dump(train_data, f)