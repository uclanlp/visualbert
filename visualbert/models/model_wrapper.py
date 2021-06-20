# Handles model training (optimizer), loading, saving

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
from utils.pytorch_misc import time_batch, save_checkpoint, clip_grad_norm, \
    restore_checkpoint, print_para, restore_best_checkpoint, load_state_dict_flexible

from visualbert.pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.DEBUG)

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

from allennlp.models import Model

class ModelWrapper():
    def __init__(self, args, train_dataset_length):
        self.scheduler = None
        self.args = args
        self.args.gradient_accumulation_steps = args.get("gradient_accumulation_steps", 1)
        self.args.fp16 = args.get("fp16", False)
        self.initialize_model(args)
        self.initialize_opimizer(args, train_dataset_length)

        self.global_step = 0
        self.called_time = 0

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def step(self, batch, eval_mode=False):
        if eval_mode:
            with torch.no_grad():
                output_dict = self.model(**batch)

                if output_dict['loss'] is not None:
                    loss = output_dict['loss'].mean()

                    output_dict['loss'] = loss

                return output_dict

        self.optimizer.zero_grad()

        output_dict = self.model(**batch)

        loss = output_dict['loss']

        cnn_loss = output_dict.get("cnn_regularization_loss", None)
        if cnn_loss is not None and self.model.module.cnn_loss_ratio != 0:
            loss = loss + cnn_loss * self.model.module.cnn_loss_ratio
            output_dict['cnn_regularization_loss'] = cnn_loss.mean().item()

        loss = loss.mean() # This is because on MultiGPU, loss is a tensor of size GPU_NUM

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.args.get("fp16", False):
            self.optimizer.backward(loss)
        else:
            loss.backward()

        if (self.called_time + 1) % self.args.gradient_accumulation_steps == 0:
            if self.args.fp16:
                # modify learning rate with special warm up BERT uses
                # if args.fp16 is False, BertAdam is used and handles this automatically
                lr_this_step = self.args.learning_rate * self.warmup_linear.get_lr(self.global_step, self.args.warmup_proportion)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr_this_step

            self.optimizer.step()
            self.global_step += 1

        self.called_time += 1

        return output_dict

    def initialize_opimizer(self, args, train_dataset_length):
        param_optimizer = list(self.model.named_parameters())
        # hack to remove pooler, which is not used
        # thus it produce None grad that break apex

        # There seems to be something that we can't 
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0} ]

        num_train_optimization_steps = int(
            train_dataset_length / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        self.num_train_optimization_steps = num_train_optimization_steps
        

        if args.get("fp16", False):
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if args.loss_scale == 0:
                self.optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                self.optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
            self.warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
                                                 t_total=num_train_optimization_steps)
        else:
            self.optimizer = BertAdam(optimizer_grouped_parameters,
                            lr=args.learning_rate,
                            warmup=args.warmup_proportion,
                            t_total=num_train_optimization_steps)

    def initialize_model(self, args):
        model = Model.from_params(vocab=None, params=Params(args.model))
        if args.get("fp16", False):
            model.half()
            print("Using FP 16, Model Halfed")
        self.model = DataParallel(model).cuda()
        

    def load_state_dict(self, state_dict_to_load):
        if isinstance(self.model, DataParallel):
            load_state_dict_flexible(self.model, state_dict_to_load["model"])
        load_state_dict_flexible(self.optimizer, state_dict_to_load["optimizer"])

    def state_dict(self):
        if isinstance(self.model, DataParallel):
            save_dict = {"model":self.model.module.state_dict(),
                     "optimizer":self.optimizer.state_dict()}
        else:
            save_dict = {"model":self.model.state_dict(),
                     "optimizer":self.optimizer.state_dict()}
        return save_dict

    def save_checkpoint(self, serialization_dir, epoch, val_metric_per_epoch, is_best = False):
        assert(serialization_dir)
        model_path = os.path.join(serialization_dir, "model_state_epoch_{}.th".format(epoch))
        model_state = self.model.module.state_dict() if isinstance(self.model, DataParallel) else self.model.state_dict()
        torch.save(model_state, model_path)

        training_state = {'epoch': epoch,
                          'val_metric_per_epoch': val_metric_per_epoch,
                          'optimizer': self.optimizer.state_dict()
                          }
        training_path = os.path.join(serialization_dir,
                                     "training_state_epoch_{}.th".format(epoch))
        torch.save(training_state, training_path)

        if is_best:
            print("Best validation performance so far. Copying weights to '{}/best.th'.".format(serialization_dir))
            shutil.copyfile(model_path, os.path.join(serialization_dir, "best.th"))

    def save_checkpoint_step(self, serialization_dir, step, epoch, is_best = False):
        
        assert(serialization_dir)
        model_path = os.path.join(serialization_dir, "model_step_{}_epoch_{}.th".format(step, epoch))
        model_state = self.model.module.state_dict() if isinstance(self.model, DataParallel) else self.model.state_dict()
        torch.save(model_state, model_path)

        training_state = {'step': step,
                          'epoch': epoch,
                          'val_metric_per_epoch': None,
                          'optimizer': self.optimizer.state_dict()
                          }
        training_path = os.path.join(serialization_dir,
                                     "training_step_{}_epoch_{}.th".format(step, epoch))
        torch.save(training_state, training_path)

    def restore_checkpoint(self, serialization_dir, epoch_to_load):
        # Restore from a training dir
        return restore_checkpoint(self.model, self.optimizer, serialization_dir, epoch_to_load)

    def restore_checkpoint_pretrained(self, restore_bin):
        # Restore from a given model path
        state_dict = torch.load(restore_bin, map_location=device_mapping(-1))
        if isinstance(self.model, DataParallel):
            model_to_load = self.model.module
        else:
            model_to_load = self.model

        own_state = model_to_load.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                print("Skipped:" + name)
                continue
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
                print("Successfully loaded: "+name)
            except:
                print("Part load failed: " + name)

    def freeze_detector(self):
        if hasattr(self.model.module, "detector"):
            detector = self.model.module.detector
            for submodule in detector.backbone.modules():
                if isinstance(submodule, BatchNorm2d):
                    submodule.track_running_stats = False
                for p in submodule.parameters():
                    p.requires_grad = False
        else:
            print("No detector found.")

    @staticmethod
    def read_and_insert_args(args, confg):
        import commentjson
        from attrdict import AttrDict
        with open(confg) as f:
            config_json = commentjson.load(f)
        dict_args = vars(args)
        config_json.update(dict_args)
        args = AttrDict(config_json)
        args.model.bert_model_name = args.bert_model_name
        return args




