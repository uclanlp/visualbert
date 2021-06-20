# coding=utf-8
# Copyleft 2019 project LXRT.

import collections
import os
import random

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
from param import args
from pretrain.lxmert_data import LXMERTDataset, LXMERTTorchDataset, LXMERTEvaluator
from pretrain.text_data import GeneralCorpusNP

from lxrt.tokenization import BertTokenizer
from lxrt.modeling import LXRTPretraining

from collections import defaultdict

DataTuple = collections.namedtuple("DataTuple", 'dataset torchdset loader evaluator vl_torchdset')
EvalDataTuple = collections.namedtuple("EvalDataTuple", 'dataset torchdset loader evaluator vl_torchdset textonly')

class TrainingMeter():
    def __init__(self):
        self.counter_dict = defaultdict(float)
        self.true_dict = defaultdict(float)

    def update(self, loss_dict):
        for key, item in loss_dict.items():
            self.counter_dict[key] += 1
            self.true_dict[key] += item

    def report(self):
        keys = list(self.counter_dict.keys())
        keys.sort()
        for key in keys:
            print("  {} : {:.7}".format(key, self.true_dict[key] / self.counter_dict[key]))
    
    def clean(self):
        self.counter_dict = defaultdict(float)
        self.true_dict = defaultdict(float)

from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter, _MultiProcessingDataLoaderIter


if args.get('random_seed', None):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.random.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)


def get_tuple(splits: str, bs: int, shuffle=False, drop_last=False, topk=-1, num_workers = 0, limit_source = [], restrict_source = None) -> DataTuple:
    # Decide which QA datasets would be used in pre-training.
    # Options: vqa, gqa, visual7w
    # Note: visual7w is a part of vgqa, we take the name here.
    qa_sets = args.qa_sets
    if qa_sets is not None:
        qa_sets = set(qa_set.lower().strip() for qa_set in qa_sets.split(","))

    # Build dataset, data loader, and evaluator.
    dset = LXMERTDataset(splits, qa_sets=qa_sets)
    tset = LXMERTTorchDataset(
        dset, 
        topk, 
        limit_source = limit_source, 
        use_visual_tag_flag = args.get("allow_tag_for_eval", False) # As this function is called for evaulation in our context
        )

    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=num_workers,
        collate_fn= tset.custom_collact_fn if args.get('custom_collact_fn', False) else lambda x: x,
        drop_last=drop_last, pin_memory=args.get("pin_memory", True)
    )
    evaluator = LXMERTEvaluator(dset)
    print()

    return DataTuple(dataset=dset, torchdset=tset, loader=data_loader, evaluator=evaluator, vl_torchdset=tset)

from lxrt.h5_data import CustomBatchSampler, ConcateDataset
def get_tuple_hybrid(splits: str, bs: int, shuffle=False, drop_last=False, num_workers=0, topk=-1, image_only_splits=None, text_only_splits = None, limit_source = [], restrict_source = None) -> DataTuple:
    # Decide which QA datasets would be used in pre-training.
    # Options: vqa, gqa, visual7w
    # Note: visual7w is a part of vgqa, we take the name here.
    qa_sets = args.qa_sets
    if qa_sets is not None:
        qa_sets = set(qa_set.lower().strip() for qa_set in qa_sets.split(","))

    # Three type of datasets: v&l, language, vision
    datasets_list_torch = []
    datasets_list = []

    if splits is not None:
        vl_dataset = LXMERTDataset(splits, qa_sets=qa_sets)
        vl_dataset_torch = LXMERTTorchDataset(vl_dataset, topk, limit_source = limit_source, randomized_pairing = args.get("randomized_pairing", False),  use_visual_tag_flag = args.get("use_visual_tag_flag", False))
        datasets_list.append(vl_dataset)
        datasets_list_torch.append(vl_dataset_torch)

    if text_only_splits is not None:
        text_only_datasets = []
        for split in text_only_splits.split("+"):
            if not("book_corpus" in split or "sbu" in split):
                text_only_dataset = LXMERTDataset(split, qa_sets=qa_sets)
                text_only_dataset_torch = LXMERTTorchDataset(text_only_dataset, topk, text_only=True, limit_source=limit_source)
                
                datasets_list.append(text_only_dataset)
                datasets_list_torch.append(text_only_dataset_torch)
                text_only_datasets.append(text_only_dataset_torch)
            else:
                text_only_dataset = None
                if "book_corpus" in split and args.get("text_shared_memory", False):
                    text_class = GeneralCorpusNP
                else:
                    #text_class = GeneralCorpus
                    pass
                text_only_dataset_torch = text_class(ann_file=args.book_corpus_path if "book_corpus" in split else args.sbu_path, pretrained_model_name="bert-base-uncased", tokenizer=None, seq_len=args.get("text_only_max_seq_len", 64), min_seq_len=args.get("text_only_min_seq_len", 64), encoding="utf-8", on_memory=True)
                datasets_list.append(text_only_dataset)
                datasets_list_torch.append(text_only_dataset_torch)
                text_only_datasets.append(text_only_dataset_torch)

    if image_only_splits is not None:
        if image_only_splits != "":
            image_only_dataset = LXMERTDataset(image_only_splits, qa_sets=qa_sets)
            image_only_dataset_torch = LXMERTTorchDataset(image_only_dataset, topk, image_only=True, use_visual_tag_flag = args.get("use_visual_tag_flag", False))
            datasets_list.append(image_only_dataset)
            datasets_list_torch.append(image_only_dataset_torch)

        if args.get("add_adhoc_google_cc_image_only", False):
            google_cc_dataset = LXMERTDataset("google_cc_train", qa_sets=qa_sets)
            google_cc_dataset_torch = LXMERTTorchDataset(google_cc_dataset, topk, image_only=True, use_visual_tag_flag=args.get("use_visual_tag_flag", False), available_split_for_cc = args.get("available_split_for_cc", [0]))
            datasets_list.append(google_cc_dataset)
            datasets_list_torch.append(google_cc_dataset_torch)
        
        if args.get("add_adhoc_open_image_image_only", False):
            open_image_dataset = LXMERTDataset("open_images_train", qa_sets=qa_sets)
            open_image_torch = LXMERTTorchDataset(open_image_dataset, topk, image_only=True, use_visual_tag_flag=args.get("use_visual_tag_flag", False))
            datasets_list.append(open_image_dataset)
            datasets_list_torch.append(open_image_torch)

    # Merge different datasets
    merged_dataset = ConcateDataset(datasets_list_torch)

    if args.task_qa:
        merged_dataset.answer_table = datasets_list[0].answer_table if datasets_list[0] is not None else None
    
    batch_sampler = CustomBatchSampler(merged_dataset.datasets, bs, upsample_ratios=args.get("upsample_ratios", [1,1,1]))
    try:
        custom_collact_fn = datasets_list_torch[0].custom_collact_fn if args.get('custom_collact_fn', False) else lambda x: x
    except:
        custom_collact_fn = datasets_list_torch[-1].custom_collact_fn if args.get('custom_collact_fn', False) else lambda x: x
    data_loader = DataLoader(
        merged_dataset, num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=custom_collact_fn,
        pin_memory=args.get("pin_memory", True)
    )
    if args.task_qa:
        evaluator = LXMERTEvaluator(datasets_list[0]) if datasets_list[0] is not None else None  # The evaluator is for task_qa so no need to have it
    else:
        evaluator = None
    print()

    if splits is not None:
        vl_torchdset = vl_dataset_torch
    else:
        vl_torchdset = datasets_list_torch[-1] # the last dataset

    return DataTuple(dataset=merged_dataset, torchdset=merged_dataset, loader=data_loader, evaluator=evaluator, vl_torchdset=vl_torchdset)

if not args.get("hybrid", False):
    train_tuple = get_tuple(args.train, args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers, limit_source = args.get("limit_source", []))
    valid_batch_size = args.get("valid_batch_size", 128)
    valid_tuple = get_tuple(args.valid, valid_batch_size, shuffle=False, drop_last=False, topk=5000, num_workers=args.get("val_num_workers", 2), limit_source = args.get("limit_source_for_val", []))
else:
    train_tuple = get_tuple_hybrid(args.train, args.batch_size, shuffle=True, num_workers = args.num_workers, drop_last=True, image_only_splits = args.train_image_only, text_only_splits = args.get("train_text_only", None), limit_source = args.get("limit_source", []))
    valid_batch_size = args.get("valid_batch_size", 128)
    
    valid_tuple = get_tuple(args.valid, valid_batch_size, shuffle=False, num_workers = args.get("val_num_workers", 2), drop_last=False, topk=5000, limit_source = args.get("limit_source_for_val", []))

from lxmert_data import symbolic_vocab

LOSSES_NAME = ('Mask_LM', 'Matched', 'Obj', 'Attr', 'Feat', 'QA')

class LXMERT:
    def __init__(self, max_seq_length):
        super().__init__()
        self.max_seq_length = max_seq_length

        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True
        )

        # Build model
        self.model = LXRTPretraining.from_pretrained(
            "bert-base-uncased",
            args = args,
            task_mask_lm=args.task_mask_lm,
            task_obj_predict=args.task_obj_predict,
            task_matched=args.task_matched,
            task_qa=args.task_qa,
            visual_losses=args.visual_losses,
            num_answers= args.num_answers if args.get("num_answers", None) else train_tuple.dataset.answer_table.num_answers
        )

        # Weight initialization and loading
        if args.from_scratch:
            print("Train from Scratch: re-initialize all BERT weights.")
            self.model.apply(self.model.init_bert_weights)

        if args.get("use_tag_symbolic_embedding", False):
            self.model.bert.embeddings.initialize_symbolic_embeddings(symbolic_vocab.get_symbolic_list(self.tokenizer))
            self.model.special_initialize_pretraining_head()
        
        if args.get("hybrid_embedding", False):
            self.model.bert.embeddings.initialize_visual_position_type_embeddings()
        
        if args.load_lxmert is not None:
            # Load lxmert would not load the answer head.
            self.load_lxmert(args.load_lxmert)
        
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model = nn.DataParallel(self.model)
        
        self.global_step = 0

    def forward(self, examples):
        
        for index, i in enumerate(examples):
            if i is not None:
                if isinstance(i, dict):
                    for key in i:
                        i[key] = (i[key][0].cuda(), i[key][1].cuda())
                else:
                    examples[index] = i.cuda()
        
        input_ids, segment_ids, input_mask, lm_labels, feats, pos, obj_labels, matched_labels, ans, visual_feats_seg_ids, visual_tags, visual_tags_mask, visual_tags_box, visual_tags_objective, visual_tags_mismatch, visual_tags_segment_ids = examples

        loss, losses, ans_logit, losses_dict = self.model(
            input_ids, segment_ids, input_mask, lm_labels,
            feats, pos, obj_labels, matched_labels, ans,
            visual_feats_seg_ids = visual_feats_seg_ids,
            visual_tags = visual_tags,
            visual_tags_mask = visual_tags_mask,
            visual_tags_box = visual_tags_box,
            visual_tags_objective = visual_tags_objective,
            visual_tags_mismatch = visual_tags_mismatch,
            visual_tags_segment_ids = visual_tags_segment_ids
        )
        return loss, losses.detach().cpu(), ans_logit, losses_dict

    def train_batch(self, optim, batch):
        
        gradient_accumulation_steps = args.get("gradient_accumulation_steps", 1)
        if (self.global_step + 1) % gradient_accumulation_steps == 0:
            optim.zero_grad()
        loss, losses, ans_logit, losses_dict = self.forward(batch)
        if args.multiGPU:
            loss = loss.mean()
            losses = losses.mean(0)
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps

        loss.backward()
        if (self.global_step + 1) % gradient_accumulation_steps == 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            optim.step()

        return loss.item(), losses.cpu().numpy(), ans_logit, losses_dict

    def valid_batch(self, batch):
        with torch.no_grad():
            loss, losses, ans_logit, losses_dict = self.forward(batch)
            if args.multiGPU:
                loss = loss.mean()
                losses = losses.mean(0)
        return loss.item(), losses.cpu().numpy(), ans_logit, losses_dict

    def train(self, train_tuple: DataTuple, eval_tuple: DataTuple):
        train_ld = train_tuple.loader

        # Optimizer
        from lxrt.optimization import BertAdam
        batch_per_epoch = len(train_ld)
        t_total = int(batch_per_epoch * args.epochs)
        warmup_ratio = args.get("warmup_ratio", 0.05)

        print("Total Iters: %d" % t_total)
        if args.get("t_total", None):
            t_total = args.t_total
            print("!! Changing to specified t_toal in args: {}".format(t_total))
        self.t_total = t_total
        warmup_iters = int(t_total * warmup_ratio)

        print("Batch per epoch: %d" % batch_per_epoch)
        print("Warm up Iters: %d" % warmup_iters)
        self.optim = BertAdam(self.model.parameters(), lr=args.lr, warmup=warmup_ratio, t_total=t_total)

        if args.load is not None:
            self.load(args.load, t_total = t_total)

        gradient_accumulation_steps = args.get("gradient_accumulation_steps", 1)
        # Train
        best_eval_loss = 9595.
        report_every = args.get("report_every", 100)

        custom_train_meter = TrainingMeter()
        
        for epoch in range(args.epochs):
            # Train
            self.model.train()
            total_loss = 0.
            total_losses = 0.
            uid2ans = {}

            for batch_id, batch in enumerate(tqdm(train_ld, total=len(train_ld))):
                if args.get("skip_training", False):
                    break

                loss, losses, logit, losses_dict = self.train_batch(self.optim, batch)
                total_loss += loss
                try:
                    total_losses += losses
                except:
                    pass

                if args.task_qa and batch[0].sent is not None:
                    assert(0) # Not used in our experiment

                    score, label = logit.max(1)
                    for datum, l in zip(batch, label.cpu().numpy()):
                        uid = datum.uid
                        ans = train_tuple.dataset.answer_table.id2ans(l)
                        uid2ans[uid] = ans
                
                for key, value in losses_dict.items():
                    losses_dict[key] = value.mean().item()  # make the losses scalar
                
                if "Masked LM" in losses_dict and losses_dict["Masked LM"] == 0:
                    del losses_dict["Masked LM"]

                custom_train_meter.update(losses_dict)

                if batch_id % report_every == 0 and batch_id > 0:
                    print("Folder: {} \n Epoch {} Iter: {}/{}".format(args.output, epoch, batch_id, len(train_ld)))
                    #print(pd.DataFrame(train_results[-report_every:]).mean())
                    custom_train_meter.report()
                    custom_train_meter.clean()
                    print()
                
                if args.get("save_step", -1) != -1 and self.global_step != 0 and (self.global_step // gradient_accumulation_steps) % args.save_step == 0:
                    self.save("Step{}".format(self.global_step))
                self.global_step += 1
            
            print("The training loss for Epoch %d is %0.4f" % (epoch, total_loss / batch_per_epoch))

            if args.task_qa:
                train_tuple.evaluator.evaluate(uid2ans, pprint=True)

            # Eval
            avg_eval_loss = self.evaluate_epoch(eval_tuple, iters=-1)

            if args.get("eval_on_train", False):
                print("On train set")
                self.evaluate_epoch(train_tuple, iters=-1)


            if avg_eval_loss < best_eval_loss:
                best_eval_loss = avg_eval_loss
                self.save("BEST_EVAL_LOSS")
            self.save("Epoch%02d" % (epoch+1))

    def evaluate_epoch(self, eval_tuple: DataTuple, iters: int=-1):
        self.model.eval()
        eval_ld = eval_tuple.loader
        total_loss = 0.
        total_losses = 0.
        uid2ans = {}
        eval_meter = TrainingMeter()
        for i, batch in enumerate(tqdm(eval_ld)):
            loss, losses, logit, losses_dict = self.valid_batch(batch)
            total_loss += loss
            try:
                total_losses += losses
            except:
                pass
            for key, value in losses_dict.items():
                losses_dict[key] = value.mean().item()
            eval_meter.update(losses_dict)

            if args.task_qa:
                score, label = logit.max(1)
                for datum, l in zip(batch, label.cpu().numpy()):
                    uid = datum.uid
                    ans = train_tuple.dataset.answer_table.id2ans(l)
                    uid2ans[uid] = ans
            if i == iters:
                break
        print("Evaluation:")
        eval_meter.report()
        print("\n\n\n\n\n\n\n\n")

        if args.task_qa:
            eval_tuple.evaluator.evaluate(uid2ans, pprint=True)

        return total_loss / len(eval_ld)
    
    def evaluate_epoch_text(self, eval_tuple: DataTuple, iters: int=-1):
        self.model.eval()
        eval_ld = eval_tuple.textonly
        total_loss = 0.
        total_losses = 0.
        uid2ans = {}
        eval_meter = TrainingMeter()
        for i, batch in enumerate(tqdm(eval_ld)):
            loss, losses, logit, losses_dict = self.valid_batch(batch)
            total_loss += loss
            total_losses += losses
            for key, value in losses_dict.items():
                losses_dict[key] = value.mean().item()
            eval_meter.update(losses_dict)

            if args.task_qa:
                score, label = logit.max(1)
                for datum, l in zip(batch, label.cpu().numpy()):
                    uid = datum.uid
                    ans = train_tuple.dataset.answer_table.id2ans(l)
                    uid2ans[uid] = ans
            if i == iters:
                break
        print("Evaluation text only:")
        eval_meter.report()
        print("\n\n\n\n\n\n\n\n")

        return total_loss / len(eval_ld)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(args.output, "%s_LXRT.pth" % name))
        
        if args.get("save_optimizer", False) and "Step" not in name:
            torch.save(self.optim.state_dict(),
                   os.path.join(args.output, "%s_LXRT_optimizer.pth" % name))
        

    def load(self, path, t_total):
        print("Load model from %s" % path)
        state_dict = torch.load("%s_LXRT.pth" % path)
        #self.model.load_state_dict(state_dict)
        from qa_answer_table import load_state_dict_flexible
        load_state_dict_flexible(self.model, state_dict)

        optimizer_path = "{}_LXRT_optimizer.pth".format(path)
        if os.path.exists(optimizer_path) and args.get("load_optimizer", True):
            print("Load optimizer from {}".format(optimizer_path))

            loaded_optim = torch.load(optimizer_path)
            if args.get("reset_schedule", False):
                for group in loaded_optim["param_groups"]:
                    group['lr'] = args.lr
                    group['warmup'] = args.warmup_ratio
                    group["t_total"] = t_total

                    for p in group['params']:
                        loaded_optim["state"][p]["step"]
                        loaded_optim["state"][p]["step"] = 0
            self.optim.load_state_dict(loaded_optim)
    

    def load_lxmert(self, path):
        print("Load LXMERT model from %s" % path)
        state_dict = torch.load("%s_LXRT.pth" % path)

        # Do not load any answer head
        for key in list(state_dict.keys()):
            if 'answer' in key:
                state_dict.pop(key)

        # Change Multi GPU to single GPU
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_state_dict[key[len("module."):]] = value
        state_dict = new_state_dict

        load_keys = set(state_dict.keys())
        model_keys = set(self.model.state_dict().keys())
        print()
        print("Keys in loaded but not in model:")
        for key in sorted(load_keys.difference(model_keys)):
            print(key)
        print()
        print("Keys in model but not in loaded:")
        for key in sorted(model_keys.difference(load_keys)):
            print(key)
        print()

        self.model.load_state_dict(state_dict, strict=False)


if __name__ == "__main__":

    lxmert = LXMERT(max_seq_length=args.get("max_seq_length", 20))


    lxmert.train(train_tuple, valid_tuple)
