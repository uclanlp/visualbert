import json
import os

def filter_examples(filename, balanced):
    with open(filename) as infile:
        original_examples = [json.loads(line) for line in infile if line]

    pair_labels = dict()
    for example in original_examples:
        urls = example["left_url"], example["right_url"]
        identifier = example["identifier"]
        label = example["label"]
        if urls not in pair_labels:
            pair_labels[urls] = list()
        pair_labels[urls].append((identifier, label))

    filtered_ids = list()
    num_appearing_more_than_once = 0
    for urls, examples in pair_labels.items():
        if len(examples) > 1:
            num_appearing_more_than_once += len(examples)
            if balanced and len(set([item[1] for item in examples])) > 1:
                for item in examples:
                    filtered_ids.append(item)
            elif not balanced and len(set(item[1] for item in examples)) == 1:
                for item in examples:
                    filtered_ids.append(item)

    print('Filtered dataset ' + str(filename) + ' with balanced=' + str(balanced))
    print('A total of %d pairs occur more than once' % num_appearing_more_than_once)
    print('Found %d valid examples' % len(filtered_ids))
    percent_true = len([example for example in filtered_ids if example[1].lower() == "true"]) / float(len(filtered_ids))
    print('Majority class: ' + '{0:.2f}'.format(100. * percent_true))

    only_ids = [item[0] for item in filtered_ids]

    bal_str = "balanced" if balanced else "unbalanced"
    with open(os.path.join(bal_str, bal_str + '_' + filename), "w") as ofile:
        for example in original_examples:
            if example["identifier"] in only_ids:
                ofile.write(json.dumps(example) + '\n')

filter_examples("dev.json", True)
filter_examples("dev.json", False)
filter_examples("test1.json", True)
filter_examples("test1.json", False)
