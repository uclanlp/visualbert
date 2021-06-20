"""
Training script. Should be pretty adaptable to whatever.
"""
import argparse
import os
import shutil
from copy import deepcopy

import multiprocessing
import numpy as np
import pandas as pd
import torch
from allennlp.common.params import Params
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.optimizers import Optimizer
from torch.nn import DataParallel
from torch.nn.modules import BatchNorm2d
from tqdm import tqdm

from allennlp.nn.util import device_mapping


from visualbert.utils.pytorch_misc import time_batch, save_checkpoint, clip_grad_norm, \
    restore_checkpoint, print_para, restore_best_checkpoint, restore_checkpoint_flexible, load_state_dict_flexible, compute_score_with_logits

from visualbert.dataloaders.vcr import VCR, VCRLoader
try:
    from visualbert.dataloaders.coco_dataset import COCODataset
except:
    print("Import COCO dataset failed.")
try:   
    from visualbert.dataloaders.nlvr_dataset import NLVRDataset
except:
    print("Import NLVR2 dataset failed.")
try:
    from visualbert.dataloaders.vqa_dataset import VQADataset
except:
    print("Import VQA dataset failed.")
try:
    from visualbert.dataloaders.flickr_dataset import Flickr30kFeatureDataset
except:
    print("Import Flickr30K dataset failed.")

from pytorch_pretrained_bert.optimization import BertAdam

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.DEBUG)

'''import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
try:
    resource.setrlimit(resource.RLIMIT_NOFILE, (40960, rlimit[1]))
    print("Setting to 40960")
except:
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))'''

from allennlp.models import Model
from visualbert.models.model_wrapper import ModelWrapper
from visualbert.models import model

#################################
from attrdict import AttrDict

parser = argparse.ArgumentParser(description='train')

parser.add_argument(
    '-folder',
    dest='folder',
    help='folder location',
    type=str,
)
parser.add_argument(
    '-no_tqdm',
    dest='no_tqdm',
    action='store_true',
)

parser.add_argument(
    '-config',
    dest='config',
    help='config location',
    type=str,
)

args = parser.parse_args()

args = ModelWrapper.read_and_insert_args(args, args.config)
##################################################### 

if os.path.exists(args.folder):
    create_flag = 0
else:
    create_flag = 1
    print("Making directories")
    os.makedirs(args.folder, exist_ok=True)

import sys
run_log_counter = 0

while(os.path.exists(args.folder + '/run_{}.log'.format(run_log_counter))):
    run_log_counter += 1

file_log = open(args.folder + '/run_{}.log'.format(run_log_counter),'w')  # File where you need to keep the logs
file_log.write("")
class Unbuffered:
    def __init__(self, stream):
       self.stream = stream
    def write(self, data):
       self.stream.write(data)
       self.stream.flush()
       file_log.write(data)    # Write the data of stdout here to a text file as well
    def flush(self):
        pass

sys.stdout = Unbuffered(sys.stdout)

NUM_GPUS = torch.cuda.device_count()
NUM_CPUS = multiprocessing.cpu_count()
if NUM_GPUS == 0:
    raise ValueError("you need gpus!")

def _to_gpu(td):
    if args.get("fp16", False):
        _to_fp16(td)

    if NUM_GPUS > 1:
        return td
    for k in td:
        if k != 'metadata':
            if td[k] is not None:
                td[k] = {k2: v.cuda(non_blocking=True) for k2, v in td[k].items()} if isinstance(td[k], dict) else td[k].cuda(non_blocking=True)
    return td
def _to_fp16(td):
    for k in td:
        if isinstance(td[k], torch.FloatTensor):
            td[k] = td[k].to(dtype=torch.float16)

num_workers = args.get("num_workers", 2)
val_workers = args.get("val_workers", 0)

TEST_DATA_READING = False
if TEST_DATA_READING:
    num_workers = 0

print(f"Using {num_workers} workers out of {NUM_CPUS} possible", flush=True)
loader_params = {'batch_size': args.train_batch_size // NUM_GPUS, 'num_gpus':NUM_GPUS, 'num_workers':num_workers}

def get_dataset_loader(args, dataset_name):
    # The VCR approach toward
    if  dataset_name == "vcr":
        train, val, test = VCR.splits(
                                  mode='rationale' if args.rationale else 'answer',
                                  only_use_relevant_dets = args.get('only_use_relevant_dets', True),
                                  do_lower_case = args.do_lower_case,
                                  bert_model_name = args.bert_model_name,
                                  max_seq_length = args.max_seq_length,
                                  pretraining = args.pretraining,
                                  pretraining_include_qa_and_qar = args.pretraining_include_qa_and_qar,
                                  complete_shuffle = args.get("complete_shuffle", False),
                                  use_alignment = args.get('use_alignment', False),
                                  add_all_features = args.add_all_features,
                                  answer_labels_path = args.get("answer_labels_path", None),
                                  vcr_annots_dir = args.vcr_annots_dir,
                                  vcr_image_dir = args.vcr_image_dir
                                  )
    elif dataset_name == "coco":
        train, val, test = COCODataset.splits(args)
    elif dataset_name == "nlvr":
        train, val, test = NLVRDataset.splits(args)
    elif dataset_name == "vqa":
        train, val, test = VQADataset.splits(args)
    elif dataset_name == "wiki":
        train, val, test = WikiDataset.splits(args)
    elif dataset_name == "flickr":
        train, val, test = Flickr30kFeatureDataset.splits(args)
    else:
        assert(0)

    loader_params = {'batch_size': args.train_batch_size // NUM_GPUS, 'num_gpus':NUM_GPUS, 'num_workers':num_workers}
    train_loader_params = deepcopy(loader_params)
    val_loader_params = deepcopy(loader_params)
    val_loader_params["num_workers"] = val_workers
    test_loader_params = deepcopy(loader_params)
    test_loader_params["num_workers"] = val_workers
    
    train_loader = VCRLoader.from_dataset(train, **train_loader_params)
    val_loader = VCRLoader.from_dataset(val, **val_loader_params)
    test_loader = VCRLoader.from_dataset(test, **test_loader_params)
    train_set_size = len(train)

    return train_loader, val_loader, test_loader, train_set_size


train_loader, val_loader, test_loader, train_set_size = get_dataset_loader(args, args.dataset)


ARGS_RESET_EVERY = args.get("print_every", 100)


train_model = ModelWrapper(args, train_set_size)

#Loading from pre-trained model
if args.restore_bin:
    train_model.restore_checkpoint_pretrained(args.restore_bin)

#Loading from previous checkpoint
if create_flag == 0:
    start_epoch, val_metric_per_epoch = train_model.restore_checkpoint(serialization_dir=args.folder, epoch_to_load = args.get("epoch_to_load", None))
    if val_metric_per_epoch is None:
        val_metric_per_epoch = []
else:
    create_flag = 1
    start_epoch, val_metric_per_epoch = 0, []

shutil.copy2(args.config, args.folder) # Always copy the config

if args.get("freeze_detector", True):
    train_model.freeze_detector()

param_shapes = print_para(train_model.model)

print(args)

print("########### Starting from {}".format(start_epoch))

num_batches = 0
    
stop_epoch = args.num_train_epochs

save_every = args.get("save_every", None)

for epoch_num in range(start_epoch, stop_epoch):
    train_results = []
    norms = []
    train_model.model.train()
    if not args.get("skip_training", False):
        for b, (time_per_batch, batch) in enumerate(time_batch(tqdm(train_loader), reset_every=ARGS_RESET_EVERY)):

            batch = _to_gpu(batch)
            
            output_dict = train_model.step(batch)

            num_batches += 1

            train_results.append(pd.Series({'loss': output_dict['loss'].mean().item(),
                                            'crl': output_dict.get("cnn_regularization_loss", 0.0),
                                            'next_sentence_loss': output_dict["next_sentence_loss"].mean().item() if "next_sentence_loss" in output_dict else 0.0,
                                            'masked_lm_loss': output_dict["masked_lm_loss"].mean().item() if "masked_lm_loss" in output_dict else 0.0,
                                            'accuracy': (train_model.model.module).get_metrics(
                                                reset=(b % ARGS_RESET_EVERY) == 0)[
                                                'accuracy'],
                                            'sec_per_batch': time_per_batch,
                                            'hr_per_epoch': len(train_loader) * time_per_batch / 3600,
                                            }))
            if b % ARGS_RESET_EVERY == 0 and b > 0:
                print("e{:2d}b{:5d}/{:5d}. \nsumm:\n{}\n~~~~~~~~~~~~~~~~~~\n".format(
                    epoch_num, b, len(train_loader),
                    pd.DataFrame(train_results[-ARGS_RESET_EVERY:]).mean(),
                ), flush=True)

            if save_every is not None and b % save_every == 0 and b != 0:
                train_model.save_checkpoint_step(args.folder, b, epoch_num)

        print("---\nTRAIN EPOCH {:2d}:\n{}\n----".format(epoch_num, pd.DataFrame(train_results).mean()))

    try:
        ### This is the eval part
        val_probs = []
        val_labels = []
        val_size = 0.0
        val_loss_sum = 0.0

        val_acc = 0.0
        val_acc_upper = 0.0
        val_instance_counter = 0.0

        val_next_sentence_loss_sum = 0.0

        train_model.eval()

        val_counter = 0

        ############ Different reporting parameters

        # for vqa, nlvr, flickr
        do_test = args.get("do_test", False) ## This one is for vqa
        if do_test:
            val_loader = test_loader
            val_dataset = val_loader.dataset
        vcr_save_result = args.get("vcr_save_result", False) # This one is for vcr

        for b, (time_per_batch, batch) in enumerate(time_batch(val_loader if args.no_tqdm else tqdm(val_loader), reset_every=ARGS_RESET_EVERY)):
            with torch.no_grad():
                batch = _to_gpu(batch)
                output_dict = train_model.step(batch, eval_mode = True)

                if not args.pretraining:
                    # Pretty clumsy code
                    if args.model.training_head_type == "vqa":
                        val_probs.append(output_dict['logits'].detach().cpu())
                        if not do_test:
                            val_labels.append(batch['label'].detach().cpu())
                    elif args.model.training_head_type == "flickr":
                        # This is because of multi-GPU
                        val_acc += (output_dict["accuracy"] * output_dict["entity_num"].float()).sum(-1).item()
                        val_acc_upper += (output_dict["upperbound_accuracy"] * output_dict["entity_num"].float()).sum(-1).item()
                        val_instance_counter += output_dict["entity_num"].sum(-1).item()

                    elif args.model.training_head_type == "multichoice":
                        val_probs.append(output_dict['logits'].detach().cpu().numpy())
                        if not do_test:
                            val_labels.append(batch['label'].detach().cpu().numpy())
                    elif args.model.training_head_type == "nlvr":
                        val_probs.append(output_dict['logits'].detach().cpu().numpy())
                        val_labels.append(batch['label'].detach().cpu().numpy())

                else:
                    val_labels.append(batch['label'].detach().cpu().numpy())

                if not do_test:
                    val_loss_sum += output_dict['loss'].mean().item() * batch['label'].size(0)
                    val_counter += batch['label'].size(0)

                    if "next_sentence_loss" in output_dict:
                        val_next_sentence_loss_sum += output_dict['next_sentence_loss'].mean().item() * batch['label'].size(0)

        if not args.pretraining:
            if args.model.training_head_type == "vqa":
                if do_test:
                    val_probs = np.concatenate(val_probs, 0)
                    val_probs = torch.Tensor(val_probs)
                    val_probs = val_probs.squeeze(1)
                    val_dataset.generate_test_file(val_probs, os.path.join(args.folder, "result.json"))
                    print("Finished testing")
                    assert(0)
                else:
                    val_labels = np.concatenate(val_labels, 0)
                    val_probs = np.concatenate(val_probs, 0)
                    val_probs = torch.Tensor(val_probs)
                    val_labels = torch.Tensor(val_labels)
                    val_probs = val_probs.squeeze(1)
                    acc = torch.sum(compute_score_with_logits(val_probs, val_labels)) / val_labels.size(0)
                    acc = acc.squeeze(-1).item()
            elif args.model.training_head_type == "flickr":
                acc = val_acc / val_instance_counter
                val_acc_upper = val_acc_upper / val_instance_counter
                print("Upper bound: {:.5f}".format(val_acc_upper))
            elif args.model.training_head_type == "multichoice": #VCR
                if not do_test:
                    val_labels = np.concatenate(val_labels, 0)
                val_probs = np.concatenate(val_probs, 0)
                if vcr_save_result:
                    if do_test:
                        file_name = "test"
                    else:
                        file_name = "val"

                    save_file_name = os.path.join(args.folder, file_name + "_qa.np")
                    if args.rationale:
                        save_file_name = os.path.join(args.folder, file_name + "_qar.np")
                    if do_test:
                        np.save(save_file_name, val_probs)
                    else:
                        np.savez(save_file_name+'z', val_probs=val_probs, val_labels=val_labels)

                        #np.save(save_file_name, (val_probs, val_labels))
                    print("Saved result to {}".format(save_file_name))
                    assert(0)

                acc = float(np.mean(val_labels == val_probs.argmax(1)))
            elif args.model.training_head_type == "nlvr":
                val_labels = np.concatenate(val_labels, 0)
                val_probs = np.concatenate(val_probs, 0)
                if args.get("report", False):
                    val_probs = val_probs.argmax(1)
                    assert(val_probs.shape[0]) == len(val_dataset)
                    result = []
                    for index, i in enumerate(val_dataset.items):
                        label = "True" if val_probs[index] == 1 else "False"
                        result.append(i["identifier"] + "," + label)
                    with open(os.path.join(args.folder, "results.csv"), "w") as f:
                        f.write("\n".join(result))
                    assert(0)
                acc = float(np.mean(val_labels == val_probs.argmax(1)))
            if not do_test:
                val_loss_avg = val_loss_sum / val_counter
                print("Val epoch {} has acc {:.5f} and loss {:.5f}".format(epoch_num, acc, val_loss_avg), flush=True)
            else:
                print("Val epoch {} has acc {:.5f}".format(epoch_num, acc), flush=True)
                assert(0)
            val_metric_per_epoch.append(acc)
        else:
            val_loss_avg = val_loss_sum / val_counter
            val_next_sentence_loss_avg = val_next_sentence_loss_sum / val_counter
            print("Val epoch {} has loss {:.5f}, next sentence loss {:.5f}".format(epoch_num, val_loss_avg, val_next_sentence_loss_avg), flush=True)
            val_metric_per_epoch.append(-val_loss_avg)
        
        if int(np.argmax(val_metric_per_epoch)) < (len(val_metric_per_epoch) - 1 - args.patience):
            print("Stopping at epoch {:2d}".format(epoch_num))
            break
        ############### Save model
        if not args.get("skip_training", False):
            train_model.save_checkpoint(args.folder, epoch_num, val_metric_per_epoch, is_best=int(np.argmax(val_metric_per_epoch)) == (len(val_metric_per_epoch) - 1))
    except KeyboardInterrupt:
        if not args.get("skip_training", False):
            train_model.save_checkpoint(args.folder, epoch_num, None, is_best=False)
        print("Something Went Wrong with Evaluation. Stopped.")
        assert(0)
    except:
        if not args.get("skip_training", False):
            train_model.save_checkpoint(args.folder, epoch_num, None, is_best=False)
        print("Something Went Wrong with Evaluation. Ignored.")
        if args.get("skip_training", False):
            assert(0)
        