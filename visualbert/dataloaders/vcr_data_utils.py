# This is adapted data/get_bert_embedding/from vcr_loader.py from R2C. Renamed to make 

import json
from collections import defaultdict
from tqdm import tqdm

from .bert_data_utils import InputExample, InputFeatures

GENDER_NEUTRAL_NAMES = ['Casey', 'Riley', 'Jessie', 'Jackie', 'Avery', 'Jaime', 'Peyton', 'Kerry', 'Jody', 'Kendall',
                        'Peyton', 'Skyler', 'Frankie', 'Pat', 'Quinn',
                        ]
##################################################################################################

def _fix_tokenization(tokenized_sent, obj_to_type, det_hist=None):
    if det_hist is None:
        det_hist = {}
    else:
        det_hist = {k: v for k, v in det_hist.items()}

    obj2count = defaultdict(int)
    # Comment this in if you want to preserve stuff from the earlier rounds.
    for v in det_hist.values():
        obj2count[v.split('_')[0]] += 1

    new_tokenization = []
    for i, tok in enumerate(tokenized_sent):
        if isinstance(tok, list):
            for int_name in tok:
                if int_name not in det_hist:
                    if obj_to_type[int_name] == 'person':
                        det_hist[int_name] = GENDER_NEUTRAL_NAMES[obj2count['person'] % len(GENDER_NEUTRAL_NAMES)]
                    else:
                        det_hist[int_name] = obj_to_type[int_name]
                    obj2count[obj_to_type[int_name]] += 1
                new_tokenization.append(det_hist[int_name])
        else:
            new_tokenization.append(tok)
    return new_tokenization, det_hist


def fix_item(item, answer_label=None, rationales=True):
    if rationales:
        assert answer_label is not None
        ctx = item['question'] + item['answer_choices'][answer_label]
    else:
        ctx = item['question']

    q_tok, hist = _fix_tokenization(ctx, item['objects'])
    choices = item['rationale_choices'] if rationales else item['answer_choices']
    a_toks = [_fix_tokenization(choice, obj_to_type=item['objects'], det_hist=hist)[0] for choice in choices]
    return q_tok, a_toks


def retokenize_with_alignment(span, tokenizer):
    tokens = []
    alignment = []
    for i, tok in enumerate(span):
        for token in tokenizer.basic_tokenizer.tokenize(tok):
            for sub_token in tokenizer.wordpiece_tokenizer.tokenize(token):
                tokens.append(sub_token)
                alignment.append(i)
    return tokens, alignment


def process_ctx_ans_for_bert(ctx_raw, ans_raw, tokenizer, counter, endingonly, max_seq_length, is_correct):
    """
    Processes a Q/A pair for BERT
    :param ctx_raw:
    :param ans_raw:
    :param tokenizer:
    :param counter:
    :param endingonly:
    :param max_seq_length:
    :param is_correct:
    :return:
    """
    context = retokenize_with_alignment(ctx_raw, tokenizer)
    answer = retokenize_with_alignment(ans_raw, tokenizer)

    if endingonly:
        take_away_from_ctx = len(answer[0]) - max_seq_length + 2
        if take_away_from_ctx > 0:
            answer = (answer[0][take_away_from_ctx:],
                      answer[1][take_away_from_ctx:])

        return InputExample(unique_id=counter, text_a=answer[0], text_b=None,
                            is_correct=is_correct), answer[1], None

    len_total = len(context[0]) + len(answer[0]) + 3
    if len_total > max_seq_length:
        take_away_from_ctx = min((len_total - max_seq_length + 1) // 2, max(len(context) - 32, 0))
        take_away_from_answer = len_total - max_seq_length + take_away_from_ctx
        context = (context[0][take_away_from_ctx:],
                   context[1][take_away_from_ctx:])
        answer = (answer[0][take_away_from_answer:],
                  answer[1][take_away_from_answer:])

        #print("FOR Q{} A {}\nLTotal was {} so take away {} from ctx and {} from answer".format(' '.join(context[0]), ' '.join(answer[0]), len_total, take_away_from_ctx,take_away_from_answer), flush=True)
        #print(len(context[0]) + len(answer[0]) + 3)

    assert len(context[0]) + len(answer[0]) + 3 <= max_seq_length

    return InputExample(unique_id=counter,
                        text_a=context[0],
                        text_b=answer[0], is_correct=is_correct), context[1], answer[1]


def data_iter(data_fn, tokenizer, max_seq_length, endingonly):
    counter = 0
    with open(data_fn, 'r') as f:
        for line_no, line in enumerate(tqdm(f)):
            item = json.loads(line)
            q_tokens, a_tokens = fix_item(item, rationales=False)
            qa_tokens, r_tokens = fix_item(item, answer_label=item['answer_label'], rationales=True)

            for (name, ctx, answers) in (('qa', q_tokens, a_tokens), ('qar', qa_tokens, r_tokens)):
                for i in range(4):
                    is_correct = item['answer_label' if name == 'qa' else 'rationale_label'] == i

                    yield process_ctx_ans_for_bert(ctx, answers[i], tokenizer, counter=counter, endingonly=endingonly,
                                                   max_seq_length=max_seq_length, is_correct=is_correct)
                    counter += 1

def data_iter_item(item, tokenizer, max_seq_length, endingonly, include_qar = False, only_qar = False):
    counter = 0
    q_tokens, a_tokens = fix_item(item, rationales=False)
    returned_list = []
    if include_qar:
        qa_tokens, r_tokens = fix_item(item, answer_label=item['answer_label'], rationales=True)
        tuples_to_process = [('qa', q_tokens, a_tokens), ('qar', qa_tokens, r_tokens)]
    elif only_qar:
        qa_tokens, r_tokens = fix_item(item, answer_label=item['answer_label'], rationales=True)
        #tuples_to_process = [('qar', qa_tokens, r_tokens)]
        tuples_to_process = [('qar', qa_tokens, r_tokens)]
    else:
        tuples_to_process = [('qa', q_tokens, a_tokens)]
    for (name, ctx, answers) in tuples_to_process:
        for i in range(4):
            try:
                is_correct = item['answer_label' if name == 'qa' else 'rationale_label'] == i
            except:
                is_correct = None
            returned_list.append(process_ctx_ans_for_bert(ctx, answers[i], tokenizer, counter=counter, endingonly=endingonly, max_seq_length=max_seq_length, is_correct=is_correct))
            counter += 1
    return returned_list



def data_iter_test(data_fn, tokenizer, max_seq_length, endingonly):
    """ Essentially this needs to be a bit separate from data_iter because we don't know which answer is correct."""
    counter = 0
    with open(data_fn, 'r') as f:
        for line_no, line in enumerate(tqdm(f)):
            item = json.loads(line)
            q_tokens, a_tokens = fix_item(item, rationales=False)

            # First yield the answers
            for i in range(4):
                yield process_ctx_ans_for_bert(q_tokens, a_tokens[i], tokenizer, counter=counter, endingonly=endingonly,
                                               max_seq_length=max_seq_length, is_correct=False)
                counter += 1

            for i in range(4):
                qa_tokens, r_tokens = fix_item(item, answer_label=i, rationales=True)
                for r_token in r_tokens:
                    yield process_ctx_ans_for_bert(qa_tokens, r_token, tokenizer, counter=counter,
                                                   endingonly=endingonly,
                                                   max_seq_length=max_seq_length, is_correct=False)
                    counter += 1
