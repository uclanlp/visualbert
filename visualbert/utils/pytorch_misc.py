"""
Question relevance model
"""

# Make stuff
import os
import re
import shutil
import time

import numpy as np
import pandas as pd
import torch
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.nn.util import device_mapping
from allennlp.training.trainer import move_optimizer_to_cuda
from torch.nn import DataParallel

import torch.nn.functional as F

def time_batch(gen, reset_every=100):
    """
    Gets timing info for a batch
    :param gen:
    :param reset_every: How often we'll reset
    :return:
    """
    start = time.time()
    start_t = 0
    for i, item in enumerate(gen):
        time_per_batch = (time.time() - start) / (i + 1 - start_t)
        yield time_per_batch, item
        if i % reset_every == 0:
            start = time.time()
            start_t = i


class Flattener(torch.nn.Module):
    def __init__(self):
        """
        Flattens last 3 dimensions to make it only batch size, -1
        """
        super(Flattener, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def pad_sequence(sequence, lengths):
    """
    :param sequence: [\sum b, .....] sequence
    :param lengths: [b1, b2, b3...] that sum to \sum b
    :return: [len(lengths), maxlen(b), .....] tensor
    """
    output = sequence.new_zeros(len(lengths), max(lengths), *sequence.shape[1:])
    start = 0
    for i, diff in enumerate(lengths):
        if diff > 0:
            output[i, :diff] = sequence[start:(start + diff)]
        start += diff
    return output


def extra_leading_dim_in_sequence(f, x, mask):
    return f(x.view(-1, *x.shape[2:]), mask.view(-1, mask.shape[2])).view(*x.shape[:3], -1)


def clip_grad_norm(named_parameters, max_norm, clip=True, verbose=False):
    """Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Variable]): an iterable of Variables that will have
            gradients normalized
        max_norm (float or int): max norm of the gradients

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    max_norm = float(max_norm)
    parameters = [(n, p) for n, p in named_parameters if p.grad is not None]
    total_norm = 0
    param_to_norm = {}
    param_to_shape = {}
    for n, p in parameters:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm ** 2
        param_to_norm[n] = param_norm
        param_to_shape[n] = tuple(p.size())
        if np.isnan(param_norm.item()):
            raise ValueError("the param {} was null.".format(n))

    total_norm = total_norm ** (1. / 2)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef.item() < 1 and clip:
        for n, p in parameters:
            p.grad.data.mul_(clip_coef)

    if verbose:
        print('---Total norm {:.3f} clip coef {:.3f}-----------------'.format(total_norm, clip_coef))
        for name, norm in sorted(param_to_norm.items(), key=lambda x: -x[1]):
            print("{:<60s}: {:.3f}, ({}: {})".format(name, norm, np.prod(param_to_shape[name]), param_to_shape[name]))
        print('-------------------------------', flush=True)

    return pd.Series({name: norm.item() for name, norm in param_to_norm.items()})


def find_latest_checkpoint(serialization_dir, epoch_to_load = None):
    """
    Return the location of the latest model and training state files.
    If there isn't a valid checkpoint then return None.
    """
    have_checkpoint = (serialization_dir is not None and
                       any("model_state_epoch_" in x for x in os.listdir(serialization_dir)))

    if not have_checkpoint:
        return None

    serialization_files = os.listdir(serialization_dir)
    model_checkpoints = [x for x in serialization_files if "model_state_epoch" in x]
    # Get the last checkpoint file.  Epochs are specified as either an
    # int (for end of epoch files) or with epoch and timestamp for
    # within epoch checkpoints, e.g. 5.2018-02-02-15-33-42
    found_epochs = [
        # pylint: disable=anomalous-backslash-in-string
        re.search("model_state_epoch_([0-9\.\-]+)\.th", x).group(1)
        for x in model_checkpoints
    ]
    int_epochs = []
    for epoch in found_epochs:
        pieces = epoch.split('.')
        if len(pieces) == 1:
            # Just a single epoch without timestamp
            int_epochs.append([int(pieces[0]), 0])
        else:
            # has a timestamp
            int_epochs.append([int(pieces[0]), pieces[1]])
    last_epoch = sorted(int_epochs, reverse=True)[0]


    if epoch_to_load is None:
        if last_epoch[1] == 0:
            epoch_to_load = str(last_epoch[0])
        else:
            epoch_to_load = '{0}.{1}'.format(last_epoch[0], last_epoch[1])

    model_path = os.path.join(serialization_dir,
                              "model_state_epoch_{}.th".format(epoch_to_load))
    training_state_path = os.path.join(serialization_dir,
                                       "training_state_epoch_{}.th".format(epoch_to_load))
    return model_path, training_state_path

def find_latest_checkpoint_step(serialization_dir, epoch_to_load = None):
    """
    Return the location of the latest model and training state files.
    If there isn't a valid checkpoint then return None.
    """
    have_checkpoint = (serialization_dir is not None and
                       any("model_step_" in x for x in os.listdir(serialization_dir)))

    if not have_checkpoint:
        return None

    serialization_files = os.listdir(serialization_dir)
    model_checkpoints = [x for x in serialization_files if "model_step_" in x]
    # Get the last checkpoint file.  Epochs are specified as either an
    # int (for end of epoch files) or with epoch and timestamp for
    # within epoch checkpoints, e.g. 5.2018-02-02-15-33-42

    info = [(x, int(x.split('_')[2]), int(x.split('_')[4].split('.')[0])) for x in model_checkpoints]

    max_epoch = -1
    max_step = -1
    max_index = -1
    for index, i in enumerate(info):
        if i[2] > max_epoch:
            max_epoch = i[2]
            max_step = i[1]
            max_index = index
        elif i[2] == max_epoch:
            if i[1] > max_step:
                max_step = i[1]
                max_index = index

    model_path = os.path.join(serialization_dir,
                              "model_step_{}_epoch_{}.th".format(max_step, max_epoch))
    training_state_path = os.path.join(serialization_dir,
                                       "training_step_{}_epoch_{}.th".format(max_step, max_epoch))
    return model_path, training_state_path


def save_checkpoint(model, optimizer, serialization_dir, epoch, val_metric_per_epoch, is_best=None,
                    learning_rate_scheduler=None) -> None:
    """
    Saves a checkpoint of the model to self._serialization_dir.
    Is a no-op if self._serialization_dir is None.
    Parameters
    ----------
    epoch : Union[int, str], required.
        The epoch of training.  If the checkpoint is saved in the middle
        of an epoch, the parameter is a string with the epoch and timestamp.
    is_best: bool, optional (default = None)
        A flag which causes the model weights at the given epoch to
        be copied to a "best.th" file. The value of this flag should
        be based on some validation metric computed by your model.
    """
    if serialization_dir is not None:
        model_path = os.path.join(serialization_dir, "model_state_epoch_{}.th".format(epoch))
        model_state = model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict()
        torch.save(model_state, model_path)

        training_state = {'epoch': epoch,
                          'val_metric_per_epoch': val_metric_per_epoch,
                          'optimizer': optimizer.state_dict()
                          }
        if learning_rate_scheduler is not None:
            training_state["learning_rate_scheduler"] = \
                learning_rate_scheduler.lr_scheduler.state_dict()
        training_path = os.path.join(serialization_dir,
                                     "training_state_epoch_{}.th".format(epoch))
        torch.save(training_state, training_path)
        if is_best:
            print("Best validation performance so far. Copying weights to '{}/best.th'.".format(serialization_dir))
            shutil.copyfile(model_path, os.path.join(serialization_dir, "best.th"))


def restore_best_checkpoint(model, serialization_dir):
    fn = os.path.join(serialization_dir, 'best.th')
    model_state = torch.load(fn, map_location=device_mapping(-1))
    assert os.path.exists(fn)
    if isinstance(model, DataParallel):
        model.module.load_state_dict(model_state)
    else:
        model.load_state_dict(model_state)

def restore_checkpoint_flexible(model, fn):
    model_state = torch.load(fn, map_location=device_mapping(-1))
    assert os.path.exists(fn)
    if isinstance(model, DataParallel):
        load_state_dict_flexible(model.module, model_state)
    else:
        load_state_dict_flexible(model, model_state)

def load_state_dict_flexible(model, state_dict):
    try:
        model.load_state_dict(state_dict)
    except:
        print("Full loading failed!! Try partial loading!!")

    own_state = model.state_dict()

    for name, param in state_dict.items():
        if name not in own_state:
            print("Skipped: " + name)
            continue
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        try:
            own_state[name].copy_(param)
            print("Successfully loaded: "+name)
        except:
            print("Part load failed: " + name)

def restore_checkpoint(model, optimizer, serialization_dir, epoch_to_load = None, learning_rate_scheduler=None):
    """
    Restores a model from a serialization_dir to the last saved checkpoint.
    This includes an epoch count and optimizer state, which is serialized separately
    from  model parameters. This function should only be used to continue training -
    if you wish to load a model for inference/load parts of a model into a new
    computation graph, you should use the native Pytorch functions:
    `` model.load_state_dict(torch.load("/path/to/model/weights.th"))``
    If ``self._serialization_dir`` does not exist or does not contain any checkpointed weights,
    this function will do nothing and return 0.
    Returns
    -------
    epoch: int
        The epoch at which to resume training, which should be one after the epoch
        in the saved training state.
    """
    latest_checkpoint = find_latest_checkpoint(serialization_dir, epoch_to_load)
    latest_checkpoint_step = find_latest_checkpoint_step(serialization_dir, epoch_to_load)

    if latest_checkpoint is None and latest_checkpoint_step is None:
        # No checkpoint to restore, start at 0
        return 0, []

    if latest_checkpoint is None:
        latest_checkpoint = latest_checkpoint_step

    model_path, training_state_path = latest_checkpoint

    # Load the parameters onto CPU, then transfer to GPU.
    # This avoids potential OOM on GPU for large models that
    # load parameters onto GPU then make a new GPU copy into the parameter
    # buffer. The GPU transfer happens implicitly in load_state_dict.
    model_state = torch.load(model_path, map_location=device_mapping(-1))
    training_state = torch.load(training_state_path, map_location=device_mapping(-1))
    if isinstance(model, DataParallel):
        model.module.load_state_dict(model_state)
    else:
        model.load_state_dict(model_state)

    # idk this is always bad luck for me
    optimizer.load_state_dict(training_state["optimizer"])

    if learning_rate_scheduler is not None and "learning_rate_scheduler" in training_state:
        learning_rate_scheduler.lr_scheduler.load_state_dict(
            training_state["learning_rate_scheduler"])
    move_optimizer_to_cuda(optimizer)

    # We didn't used to save `validation_metric_per_epoch`, so we can't assume
    # that it's part of the trainer state. If it's not there, an empty list is all
    # we can do.
    if "val_metric_per_epoch" not in training_state:
        print("trainer state `val_metric_per_epoch` not found, using empty list")
        val_metric_per_epoch: []
    else:
        val_metric_per_epoch = training_state["val_metric_per_epoch"]

    if isinstance(training_state["epoch"], int):
        epoch_to_return = training_state["epoch"] + 1
    else:
        epoch_to_return = int(training_state["epoch"].split('.')[0]) + 1

    print("########### Restroing states... from {}, at epoch {}".format(model_path, epoch_to_return))
    if "step" in training_state:
        print("########### Restroing states... from {}, at step {}".format(model_path, training_state["step"]))

    return epoch_to_return, val_metric_per_epoch

def detokenize(array, vocab):
    """
    Given an array of ints, we'll turn this into a string or a list of strings.
    :param array: possibly multidimensional numpy array
    :return:
    """
    if array.ndim > 1:
        return [detokenize(x, vocab) for x in array]
    tokenized = [vocab.get_token_from_index(v) for v in array]
    return ' '.join([x for x in tokenized if x not in (vocab._padding_token, START_SYMBOL, END_SYMBOL)])


def print_para(model):
    """
    Prints parameters of a model
    :param opt:
    :return:
    """
    st = {}
    total_params = 0
    total_params_training = 0
    for p_name, p in model.named_parameters():
        # if not ('bias' in p_name.split('.')[-1] or 'bn' in p_name.split('.')[-1]):
        st[p_name] = ([str(x) for x in p.size()], np.prod(p.size()), p.requires_grad)
        total_params += np.prod(p.size())
        if p.requires_grad:
            total_params_training += np.prod(p.size())
    pd.set_option('display.max_columns', None)
    shapes_df = pd.DataFrame([(p_name, '[{}]'.format(','.join(size)), prod, p_req_grad)
                              for p_name, (size, prod, p_req_grad) in sorted(st.items(), key=lambda x: -x[1][1])],
                             columns=['name', 'shape', 'size', 'requires_grad']).set_index('name')

    print('\n {:.1f}M total parameters. {:.1f}M training \n ----- \n {} \n ----'.format(total_params / 1000000.0,
                                                                                        total_params_training / 1000000.0,
                                                                                        shapes_df.to_string()),
          flush=True)
    return shapes_df


def batch_index_iterator(len_l, batch_size, skip_end=True):
    """
    Provides indices that iterate over a list
    :param len_l: int representing size of thing that we will
        iterate over
    :param batch_size: size of each batch
    :param skip_end: if true, don't iterate over the last batch
    :return: A generator that returns (start, end) tuples
        as it goes through all batches
    """
    iterate_until = len_l
    if skip_end:
        iterate_until = (len_l // batch_size) * batch_size

    for b_start in range(0, iterate_until, batch_size):
        yield (b_start, min(b_start + batch_size, len_l))


def batch_iterator(seq, batch_size, skip_end=True):
    for b_start, b_end in batch_index_iterator(len(seq), batch_size, skip_end=skip_end):
        yield seq[b_start:b_end]

def masked_unk_softmax(x, dim, mask_idx):
    x1 = F.softmax(x, dim=dim)
    x1[:, mask_idx] = 0
    x1_sum = torch.sum(x1, dim=1, keepdim=True)
    y = x1 / x1_sum
    return y

def compute_score_with_logits(logits, labels):
    logits = masked_unk_softmax(logits, 1, 0)
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros_like(labels)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores