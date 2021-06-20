import json
import numpy as np
import sys

# Preds file
preds = dict()
with open(sys.argv[1]) as infile:
    for line in infile:
        identifier, assignment = line.strip().split(',')
        preds[identifier] = assignment

# Annotations file
sent_to_annot = dict()
categories = set()

with open(sys.argv[2]) as infile:
    examples = [example for example in infile.read().split('\n\n') if example.strip()]
    print('Loaded %d annotated examples.' % len(examples))

    for example in examples:
        lines = example.split('\n')
        sent = lines[0]
        sent_to_annot[sent] = list()
        for category in lines[1:]:
            category = category[2:]
            sent_to_annot[sent].append(category)
            categories.add(category)

print('Found %d categories.' % len(categories))
category_corrects = dict()
for category in categories:
    category_corrects[category] = list()

# Labels file
with open(sys.argv[3]) as infile:
    for line in infile:
        example = json.loads(line)
        identifier = example["identifier"]
        label = example["label"].lower()
        sentence = example["sentence"]

        assignment = preds[identifier].lower()

        if assignment in {'true', 'false'}:
            correct = int(assignment == label)
        else:
            raise ValueError('Assignment is not true/false: ' + assignment)
        
        if sentence in sent_to_annot:
            categories = sent_to_annot[sentence]

            for category in categories:
                category_corrects[category].append(correct)

print('Per-category accuracy:')
for category, corrects in sorted(category_corrects.items(), key = lambda x: x[0]):
    print(category + ': ' + '{0:.2f}'.format(100. * np.mean(np.array(corrects))) + ' (of %d examples)' % len(corrects))
