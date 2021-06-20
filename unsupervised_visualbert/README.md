
## Intro
This is code for the paper Unsupervised Vision-and-Language Pre-training Without Parallel Images and Captions (NAACL 2021) ([Link](https://arxiv.org/abs/2010.12831)).

The code is adopted from both the original VisualBERT code and [LXMERT](https://github.com/airsplay/lxmert).  Many thanks to Hao Tan for developing the great LXMERT codebase and hosting some data files!

## Data & Files Required

### Pre-training Data
1. Vocabulary files for the [BUTD](https://github.com/peteanderson80/bottom-up-attention) detector

    ``` bash
    mkdir -p data/vocabs/
    wget https://raw.githubusercontent.com/peteanderson80/bottom-up-attention/master/data/genome/1600-400-20/attributes_vocab.txt -P data/vocabs/
    wget https://raw.githubusercontent.com/peteanderson80/bottom-up-attention/master/data/genome/1600-400-20/objects_vocab.txt -P data/vocabs/
    wget https://raw.githubusercontent.com/peteanderson80/bottom-up-attention/master/data/genome/1600-400-20/relations_vocab.txt -P data/vocabs/
    ```

2. Pre-training caption files
    Download the caption files from [LXMERT](https://github.com/airsplay/lxmert):

    ```bash
    mkdir -p data/lxmert
    wget nlp.cs.unc.edu/data/lxmert_data/lxmert/mscoco_train.json -P data/lxmert/
    wget nlp.cs.unc.edu/data/lxmert_data/lxmert/mscoco_nominival.json -P data/lxmert/
    wget nlp.cs.unc.edu/data/lxmert_data/lxmert/vgnococo.json -P data/lxmert/
    wget nlp.cs.unc.edu/data/lxmert_data/lxmert/mscoco_minival.json -P data/lxmert/
    ```

3. COCO/VG Image Features
    In the paper, we used Conceptual Captions for pre-training. But the image features take up more than 800G so we cannot publisize the image features for now. Instead, we provide scripts to run on COCO/VG images and COCO/VG captions. First download the image features files from [LXMERT](https://github.com/airsplay/lxmert):

    MSCOCO features:

    ```bash
    mkdir -p data/mscoco_imgfeat
    wget nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/train2014_obj36.zip -P data/mscoco_imgfeat
    unzip data/mscoco_imgfeat/train2014_obj36.zip -d data/mscoco_imgfeat && rm data/mscoco_imgfeat/train2014_obj36.zip
    wget nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/val2014_obj36.zip -P data/mscoco_imgfeat
    unzip data/mscoco_imgfeat/val2014_obj36.zip -d data && rm data/mscoco_imgfeat/val2014_obj36.zip
    ```

    VG features:

    ```bash
    mkdir -p data/vg_gqa_imgfeat
    wget nlp.cs.unc.edu/data/lxmert_data/vg_gqa_imgfeat/vg_gqa_obj36.zip -P data/vg_gqa_imgfeat
    unzip data/vg_gqa_imgfeat/vg_gqa_obj36.zip -d data && rm data/vg_gqa_imgfeat/vg_gqa_obj36.zip
    ```

    Then run the script to convert them into HDF5 format (for faster reading):

    ```bash
    python tools/convert_tsv_to_h5.py
    ```

4. BookCorpus

    We got our version of the BookCorpus from [VL-BERT](https://github.com/jackroos/VL-BERT/blob/master/data/PREPARE_DATA.md). After downloading the [file](https://drive.google.com/file/d/16T5EYqIjO-tAj1OFxz6bnnzEABCusCcv/view), please put it under `data/lxmert/` as `bc1g.doc`.



### VQA Data
1. Download the annotation files from [LXMERT](https://github.com/airsplay/lxmert)

    ```bash
    mkdir -p data/vqa
    wget nlp.cs.unc.edu/data/lxmert_data/vqa/train.json -P data/vqa/
    wget nlp.cs.unc.edu/data/lxmert_data/vqa/nominival.json -P  data/vqa/
    wget nlp.cs.unc.edu/data/lxmert_data/vqa/minival.json -P data/vqa/
    ```

2. COCO/VG Image Features. Please refer to instructions in downloading pre-training data.

## Environment Setup

I recommend using docker to run the experiments. Use the image `pytorch/pytorch:1.4-cuda10.1-cudnn7-devel` as a start. 

```bash
pip install yacs easydict pycocotools matplotlib pillow commentjson attrdict boto3 h5py requests scikit-learn ftfy regex tqdm ml_collections msgpack lz4 msgpack_numpy lmdb pandas
conda install --yes -c pytorch torchvision cudatoolkit=10.1 pytorch=1.4.0
```

## Pre-training
Below is an example config that conducts pre-training on COCO. As the image features of Conceptual Captions take up more than 800G so we cannot publisize the image features for now. The config used to train on Conceptual Captions is in `configs/pretrain/conceptual_captions.json`.

Command:
```bash
export PYTHONPATH=$PYTHONPATH:src
CUDA_VISIBLE_DEVICES=0 python src/pretrain/lxmert_pretrain.py --multiGPU --output ./snap/test --config ./configs/pretrain/unsupervised.json
```
Model checkpoint trained on Conceptual Captions ([GoogleDrive](https://drive.google.com/file/d/1j9hxN_Ky1S3zPuiqpDdfoHB_LJa5sMSZ/view?usp=sharing)).

Caveats: in order to do memory-efficient training, we used shared memory array among processes. So please delete any file under `/dev/shm/` with a prefix of `sharearray_`. (This is partially the reason we recommand docker, as other people (though highly unlikely) may also be using shared memory array with the same name.)

## Fine-tuning

We provide the command to run on VQA.

1. Training

    Download the pre-trained checkpoint as in the previous section and save it as `snap/pretrain/CC_Unsupervised_LXRT.pth`.

    ```bash
    export PYTHONPATH=$PYTHONPATH:src
    CUDA_VISIBLE_DEVICES=0 python src/tasks/vqa.py --multiGPU --output ./snap/vqa_test --config ./configs/vqa.json
    ```

2. Testing on minival

    Download the pre-trained checkpoint ([GooglDrive](https://drive.google.com/file/d/1lrGQUjnFOTK8U6Qi9BYMhEEHa23ue1KL/view?usp=sharing)) and save it as `snap/vqa.pth`.

    ```bash
    export PYTHONPATH=$PYTHONPATH:src
    CUDA_VISIBLE_DEVICES=0 python src/tasks/vqa.py --multiGPU --output ./snap/vqa_test --config ./configs/vqa.json --test val --load snap/vqa
    ```
    This should give the score `0.6807`.

3. Testing on test.

    ```bash
    export PYTHONPATH=$PYTHONPATH:src
    CUDA_VISIBLE_DEVICES=0 python src/tasks/vqa.py --multiGPU --output ./snap/vqa_test --config ./configs/vqa.json --test test --load snap/vqa
    ```
    The file `vqa_test/test_predict.json` could be submitted to the official VQA leaderboard. The model we provide should give a score on test-dev close to 70.7.