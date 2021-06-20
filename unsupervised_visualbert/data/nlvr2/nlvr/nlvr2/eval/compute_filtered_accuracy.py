import json
import numpy as np
import sys

# Preds file
preds = dict()
with open(sys.argv[1]) as infile:
    for line in infile:
        identifier, assignment = line.strip().split(',')
        preds[identifier] = assignment

# Labels file
corrects = list()
with open(sys.argv[2]) as infile:
    for line in infile:
        example = json.loads(line)
        identifier = example["identifier"]
        label = example["label"].lower()

        assignment = preds[identifier].lower()
        if assignment in {'true','false'}:
            corrects.append(int(assignment == label))
        else:
            print(assignment)
print(100. * np.mean(np.array(corrects)))
