# Cornell Natural Language for Visual Reasoning for Real (NLVR2)

Website: http://lic.nlp.cornell.edu/nlvr/

The corpus and task are described in: A corpus for reasoning about natural language grounded in photographs. Alane Suhr, Stephanie Zhou, Ally Zhang, Iris Zhang, Huajun Bai, and Yoav Artzi. To appear in ACL 2019, https://arxiv.org/abs/1811.00491.

## Repository structure
The `data` directory contains JSON files representing the training, development, and public test sets. The `util` directory contains scripts for downloading the images, as well as hashes for all images. The `eval` directory contains scripts for evaluating your models on the data and computing both accuracy and consistency.

## JSON files
Each line includes one example, represented as a JSON object. The critical fields are:

* `sentence`: The natural language sentence describing the pair of images for this example.
* `left_url`: The URL of the left image in the pair.
* `right_url`: The URL of the right image in the pair.
* `label`: The label: true or false.
* `identifier`: The unique identifier for the image, in the format: `split-set_id-pair_id-sentence-id`. `split` is the split of the data (train, test, or development). `set_id` is the unique identifier of the original eight-image set used in the sentence-writing task. `pair_id` indicates which of the pairs in the set it corresponds to (and is between 0 and 3). `sentence-id` indicates which of the sentences is associated with this pair (and is either 0 or 1 -- each image pair is associated with at most two sentences).

Some other useful fields are:
* `writer`: The (anonymized) identifier of the worker who wrote the sentence. The identifiers are the same across splits of the data.
* `validation`: The initial validation judgment, including the anonymized worker ID and their judgment.
* `extra_validations`: In the development and test sets, this is the set of extra judgments acquired for each example, including the anonymized worker ID and their judgment.
* `synset`: The synset associated with the example.
* `query`: The query used to find the set of images. You can ignore the numbers suffixing the query; these uniquely identify image sets for each query. 
* `directory`: In the train set, this represents the assigned directory for each example. There are 100 directories in total, and unique image pairs do not appear in multiple directories. This means you can easily sample a validation set from a subset of directories.

`test1.json` includes the public test set.

We assume a consistent naming of the image files associated with each example. Given the identifier `split-set_id-pair_id-sentence-id`, the left and right images are named `split-set_id-pair_id-img0.png` and `split-set_id-pair_id-img1.png` respectively. Despite the extension of `.png`, not all images are actually PNGs. However, most image displaying software as well as libraries like PIL will process images according to the headers of the files themselves, rather than the extension. We only ran into problems using the default file browser in Ubuntu, and instead used imagemagick to browse images on that platform.  

## Balanced and unbalanced subsets
We provide JSON files containing subsets of the development and test sets where unique image pairs appear multiple times with *balanced* or *unbalanced* labels. For discussion of these subsets, and visual bias in NLVR2, see [this notebook](http://lil.nlp.cornell.edu/nlvr/NLVR2BiasAnalysis.html). The subsets were generated using the script `data/filter_data.py`, and the directories `data/balanced` and `data/unbalanced` respectively contain the balanced and unbalanced subsets.

## Downloading the images
The script `util/download_images.py` will download images from the URLs included in the JSON files. It takes three arguments: the path of the JSON file, a directory to which the images will be saved, and the path of the hash file (hash files are included in `util/hashes/`). For each image, the script sends a request to the URL, as long as the image is not downloaded yet, and compares it with the saved hash. We use the `imagehash` library for comparing the hashes. There is a timeout of two seconds for downloading. In addition, the script should catch any errors with accessing the URL. 

If the image could not be downloaded, it saves the identifier to a file (`*_failed_imgs.txt`). If the hash was not expected, it saves the identifier to a different file (`*_failed_hashes.txt`). For each image attempted, it saves it to a file (`*_checked_imgs.txt`). This allows you to stop and restart the download without going over images that have already been checked.

In total, the download can take a long time. This script took about a day to run on the development set. In addition, because the data was collected over the course of a year, some URLs are no longer accessible. We estimate that about 5% of the data is inaccessible from the saved URLs.

## Direct image download
We do not own copyright for the images included in the dataset. Thus, we cannot share the images publicly. However, we can provide direct access to the images as long as you are using them for research purposes. To obtain access, please fill out the linked [Google Form]( https://goo.gl/forms/yS29stWnFWzrDBFH3). This form asks for your basic information and asks you to agree to our Terms of Service. We will get back to you within a week. If you have any questions, please email `nlvr@googlegroups.com`.

## Evaluation scripts
To measure both accuracy (precision) and consistency on your predictions, use the `eval/metrics.py` script. This assumes that your predictions will be in a CSV format, with the first value as the example's unique identifier and the second as the prediction (in the same format as labels in the JSON files). It will give an error if predictions are missing or received more predictions than it expected. To run, use the following:

```python eval/metrics.py PATH_TO_YOUR_CSV_PREDICTIONS.csv data/EVAL_SPLIT.json```

To compute the accuracy across a subset of the development or test data (e.g., the balanced development set), use the `compute_filtered_accuracy.py` script like so:

```python eval/compute_filtered_accuracy.py PATH_TO_YOUR_CSV_PREDICTIONS.csv data/SUBSET_NAME/EVAL_SPLIT.json```

where `SUBSET_NAME` is `balanced` or `unbalanced`.

We also provide the 800 unique development sentences comprising 2868 examples annotated for linguistic phenomena. These annotations are present in `util/annotated_dev_examples.txt`. See below for a description of the phenomena. To evaluate your model's performance on each phenomenon, use the `eval/compute_category_accuracy.py` script like so:

```python eval/compute_category_accuracy.py PATH_TO_YOUR_CSV_PREDICTIONS.csv util/annotated_dev_examples.txt data/dev.json```.

## Linguistic phenomena
We provide the 800 unique development sentences that appear in 2868 examples annotated for linguistic phenomena in `util/annotated_dev_examples.txt`. We consider thirteen linguistic phenomena (described in the order they appear in the paper):

* **Cardinality (hard)** includes references to exact counts of objects. E.g., ***Six** rolls of paper towels...*
* **Cardinality (soft)** includes references to bounds or ranges on counts of objects. E.g., ***No more than two** cheetahs are present.*
* **Existential quantifiers** explicitly state the existence of an object. E.g., ***There are** at most 3 water buffalos...*
* **Universal quantifiers** require all objects in its scope to satisfy some proposition described in the sentence. E.g., *... a line of fence posts with one large darkly colored bird on top of **each** post.*
* **Coordination** includes linguistic conjunction (e.g., *and*) and disjunction (e.g., *or*). E.g., *Each image contains only one wolf, **and** all images...*
* **Coreference** includes pronominal reference to an object mentioned in elsewhere in the statement. E.g., *... animals very close to **each other**...*
* **Spatial relations** includes descriptions of a physical relationship between two objects, e.g., *A stylus is **near** a laptop...*
* **Comparative** includes comparing properties of objects or sets of objects (including between images). E.g., *There are **more** birds in the image on the right **than** in the image on the left.*
* **Presupposition** introduces objects or propositions which introduce implicit assumptions about the image pair (see [Wikipedia on presupposition](https://en.wikipedia.org/wiki/Presupposition) for a discussion). We consider presupposition if it presupposes something that isn't true about the world in general (e.g., the presence of the sky). An example from NLVR2 is *A cookie sits in **the dessert** in the image on the left.*
* **Negation** includes negating propositions, adjectives, or adverbs in the statement. E.g., *The front paws ... are **not** touching the ground.*
* We also include three categories of syntactic attachment ambiguity. These include CC (coordination) ambiguity (e.g., *The left image shows a cream-layered dessert in a footed clear glass which includes sliced peanut butter cups **and** brownie chunks*, where *and* could attach either to *shows* or *includes*), PP (prepositional) ambiguity (e.g., *At least one panda is sitting near a fallen branch **on the ground**.*, where *on* could attach to either *sitting* or *branch*), and SBAR (subordinating conjunction) ambiguity (e.g., *Balloons float in a blue sky with dappled clouds on strings **that** angle rightward...*, where *that* could attach to *balloons*, *strings*, or more unlikely *clouds*). 

## Running on the leaderboard held-out test set
We require **two months or more** between runs on the leaderboard test set. We will do our best to run within two weeks (usually we will run much faster). We will only post results on the leaderboard when an online description of the system is available. Testing on the leaderboard test set is meant to be the final step before publication. Under extreme circumstances, we reserve the right to limit running on the leaderboard test set to systems that are mature for publication. 

We don't provide the unreleased test inputs publicly -- you will need to send your model code and scripts for inference. Your model should generate a prediction file in the format specified above (under "evaluation scripts"). 

## Note about sampling a validation set
The training set contains many examples which use the same initial set of eight images. When selecting a validation set to use, we suggest enforcing that each unique image set does not appear in both the validation set and the training set used to update model parameters. *Update 21 Dec. 2018:* We bucketed each example in the training set into one of 100 buckets, ensuring that initial unique sets do not appear across buckets. We suggest that you compose your validation set as a subset of these buckets. 


### Thanks!
This research was supported by the NSF (CRII-1656998), a Facebook ParlAI Research Award, an AI2 Key Scientific Challenges Award, Amazon Cloud Credits Grant, and support from Women in Technology New York. This material is based on work supported by the National Science Foundation Graduate Research Fellowship under Grant No. DGE-1650441.  We thank Mark Yatskar and Noah Snavely for their comments and suggestions, and the workers who participated in our data collection for their contributions.
