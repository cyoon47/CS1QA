# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BERT classification fine-tuning: utilities to work with GLUE tasks """

from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
import jsonlines
from io import open
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
from queue import PriorityQueue

csv.field_size_limit(sys.maxsize)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple span prediction."""

    def __init__(self, guid, labNo, taskNo, questioner, question, code, startLine, endLine):
        """Constructs a InputExample.

        Args:
            guid: Unique ID
            labNo: Lab number of the question.
            taskNo: Task number of the question.
            questioner: Person asking the question, either student or TA
            question: Question text
            code: The code at the time the question was asked
            label: The label of the example.
        """
        self.guid = guid
        self.labNo = labNo
        self.taskNo = taskNo
        self.questioner = questioner
        self.question = question
        self.code = code
        self.startLine = startLine
        self.endLine = endLine


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, start_index, end_index, start_line, b_start_index, b_end_index, example_id=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_index = start_index
        self.end_index = end_index
        self.start_line = start_line
        self.b_start_index = b_start_index
        self.b_end_index = b_end_index
        self.example_id = example_id


class DataProcessor(object):
    """Base class for data converters for span prediction data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_jsonl(cls, input_file):
        with jsonlines.open(input_file, 'r') as f:
            lines = [l for l in f]
        return lines

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            lines = []
            for line in f.readlines():
                line = line.strip().split('<CODESPLIT>')
                if len(line) != 5:
                    continue
                lines.append(line)
            return lines


class SpanSelectionProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir, train_file):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, train_file)))
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, train_file)), "train")

    def get_dev_examples(self, data_dir, dev_file):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, dev_file)))
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, dev_file)), "dev")

    def get_test_examples(self, data_dir, test_file, test_example_begin, test_example_end):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, test_file)))
        logger.info("test_example_begin: {}, test_example_end: {}".format(test_example_begin, test_example_end))
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, test_file)), "test", test_example_begin, test_example_end)

    def get_labels(self):
        """See base class."""
        return ["code_understanding", "logical", "error", "usage", "algorithm", "task", "comparison", "reasoning", "code_explain", "variable", "guiding"]

    def _create_examples(self, lines, set_type, begin=0, end=-1):
        """Creates examples for the training and dev sets."""
        # guid, labNo, taskNo, questioner, question, code, label
        examples = []
        example_lines = lines[begin:end]
        for (i, line) in enumerate(example_lines):
            guid = "%s-%s" % (set_type, i)
            labNo = line['labNo']
            taskNo = line['taskNo']
            questioner = line['questioner']
            question = line['question']
            code = line['code']

            startLine = line['startLine']
            endLine = line['endLine']

            examples.append(
                InputExample(guid=guid, labNo=labNo, taskNo=taskNo, questioner=questioner,\
                            question=question, code=code, startLine=startLine, endLine=endLine))
            # if i == 4:
            #     break
        if (set_type == 'test'):
            return examples, lines
        else:
            return examples


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True,
                                 model_type):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    if model_type == 'roberta':
        double_line_token, newline_token = tokenizer.tokenize("a\n\n\n")[1:]
    elif model_type == 'xlm':
        double_line_token, newline_token = None, tokenizer.tokenize("a\n")[1]

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    print_log = False

    max_q_length = 0

    for example in examples:
        text_a = ' '.join([str(example.labNo), str(example.taskNo), example.questioner, example.question])
        tokens_a = tokenizer.tokenize(text_a)
        max_q_length = max(max_q_length, len(tokens_a))

    for (ex_index, example) in enumerate(examples):
        if print_log and ex_index != 1546:
            continue
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        if print_log: logger.info("index: %d" % ex_index)
        
        # labNo, taskNo, questioner, question, code, label
        text_a = ' '.join([str(example.labNo), str(example.taskNo), example.questioner, example.question])
        # text_a = ' '.join([str(example.labNo), str(example.taskNo), example.questioner])
        tokens_a = tokenizer.tokenize(text_a)#[:50]
        # tokens_a = tokens_a + ["PAD"] * (max_q_length - len(tokens_a))
        # logger.info("tokens_a: %s" % tokens_a)

        tokens_b = None
        if example.code:
            tokens_b = tokenizer.tokenize(example.code)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            
            # _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            tokens_b_split = _split_long_tokens(tokenizer, tokens_a, tokens_b, max_seq_length - 4, model_type)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
        
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.

        # tokens = tokens_a + [sep_token]
        tokens = tokens_a + [sep_token, sep_token]
        # logger.info("tokens: %s" % tokens)

        # logger.info("tokens_a length: %d" % len(tokens_a))
        segment_ids = [sequence_a_segment_id] * len(tokens)
        all_tokens = []
        all_segments = []
        if print_log: logger.info("Split length: %d" % len(tokens_b_split))
        if print_log: logger.info("Start Line: %s, End Line: %s" % (str(example.startLine), str(example.endLine)))
        if tokens_b:
            for i, tok_b in enumerate(tokens_b_split):
                # logger.info("Code: %s", tokenizer.convert_tokens_to_string(tok_b[0]))
                curr_token = tokens[:]
                
                b_start_index = len(curr_token)
                b_end_index = b_start_index + len(tok_b[0]) - 1

                curr_segment = segment_ids[:]
                # all_tokens.append(curr_token + tok_b + [sep_token])
                # all_segments.append(curr_segment + [sequence_b_segment_id] * (len(tok_b) + 1))
                curr_line_start = tok_b[1]
                curr_token += tok_b[0] + [sep_token]
                if print_log: logger.info("Token at b_start_index (%d): %s" % (b_start_index, curr_token[b_start_index]))
                curr_segment += [sequence_b_segment_id] * (len(tok_b[0]) + 1)

                if cls_token_at_end:
                    curr_token = curr_token + [cls_token]
                    curr_segment = curr_segment + [cls_token_segment_id]
                else:
                    curr_token = [cls_token] + curr_token
                    curr_segment = [cls_token_segment_id] + curr_segment
                    b_start_index += 1
                    b_end_index += 1
                
                input_ids = tokenizer.convert_tokens_to_ids(curr_token)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

                # Zero-pad up to the sequence length.
                padding_length = max_seq_length - len(input_ids)
                if pad_on_left:
                    input_ids = ([pad_token] * padding_length) + input_ids
                    input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                    curr_segment = ([pad_token_segment_id] * padding_length) + curr_segment
                else:
                    input_ids = input_ids + ([pad_token] * padding_length)
                    input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                    curr_segment = curr_segment + ([pad_token_segment_id] * padding_length)

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(curr_segment) == max_seq_length


                start_position, end_position = None, None
                if print_log: logger.info("curr_line_start: %d" % curr_line_start)
                if example.startLine == None or curr_line_start > example.endLine:
                    start_position, end_position = 0, 0
                else:
                    new_line_lists = []
                    lines_in_snippet = 0
                    prev_end_line_index = -1
                    for idx, x in enumerate(tok_b[0]):
                        if x == newline_token:
                            new_line_lists.append((prev_end_line_index+1, idx, 1))
                            lines_in_snippet += 1
                            prev_end_line_index = idx
                        elif x == double_line_token:
                            new_line_lists.append((prev_end_line_index+1, idx, 2))
                            lines_in_snippet += 2
                            prev_end_line_index = idx
                    
                    if print_log: logger.info("new_line_lists: %s" % new_line_lists)

                    if len(new_line_lists) != 0 and new_line_lists[-1][1] != len(tok_b[0]) - 1: # Does not end with newline token
                        if print_log: logger.info("Does not end!!!!!!!!!!")
                        lines_in_snippet += 1

                    if print_log: logger.info("lines_in_snippet: %d" % lines_in_snippet)
                    if print_log: logger.info("last line: %s" % tokenizer.convert_tokens_to_string(tok_b[0][new_line_lists[-1][0]:new_line_lists[-1][1]+1]))

                    if curr_line_start + lines_in_snippet <= example.startLine:
                        start_position, end_position = 0, 0
                    else:
                        curr_line = curr_line_start
                        assert len(new_line_lists) > 0
                        for idx, x in enumerate(new_line_lists):
                            if curr_line >= example.startLine and start_position == None:
                                start_position = x[0] + b_start_index
                            if curr_line >= example.endLine:
                                end_position = x[1] + b_start_index
                                break
                            
                            curr_line += x[2]

                            if idx == len(new_line_lists)-1 and curr_line >= example.endLine and new_line_lists[-1][1] != len(tok_b[0]) - 1:
                                start_position = x[1] + b_start_index + 1

                if start_position != None and end_position == None:
                    if print_log: logger.info("Overflow!!")
                    end_position = b_end_index
                
                assert start_position != None and end_position != None

                if output_mode == "classification":
                    label_id = label_map[example.label]
                elif output_mode == "regression":
                    label_id = float(example.label)
                elif output_mode == "questionanswering":
                    pass
                else:
                    raise KeyError(output_mode)

                # if ex_index < 5:
                if print_log or ex_index == 0:
                    logger.info("*** Example ***")
                    logger.info("guid: %s" % (example.guid))
                    # logger.info("tokens: %s" % " ".join(
                    #     [str(x) for x in curr_token]))
                    # logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                    # logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                    # logger.info("segment_ids: %s" % " ".join([str(x) for x in curr_segment]))
                    logger.info("start_pos: %d, end_pos: %d" % (start_position, end_position))
                    logger.info("Question: %s" % tokenizer.convert_tokens_to_string(curr_token[1:b_start_index-1]))
                    # logger.info("all tokens:\n%s\n" % tokenizer.convert_tokens_to_string(curr_token))
                    # logger.info("Curr_token length: %d" % len(curr_token))
                    # logger.info("Token at start_pos: %s" % tokenizer.convert_tokens_to_string(curr_token[start_position]))
                    # logger.info("Token at end_pos: %s" % tokenizer.convert_tokens_to_string(curr_token[end_position]))
                    # logger.info("selected tokens:\n%s\n" % tokenizer.convert_tokens_to_string(curr_token[start_position:end_position+1]))

                features.append(
                    InputFeatures(input_ids=input_ids,
                                input_mask=input_mask,
                                segment_ids=curr_segment,
                                start_index=start_position,
                                end_index=end_position,
                                start_line=curr_line_start,
                                b_start_index=b_start_index,
                                b_end_index=b_end_index,
                                example_id=ex_index))
            if print_log: sys.exit()
    return features


def _split_long_tokens(tokenizer, tokens_a, tokens_b, max_length):
    """When total token length is greater than the maximum length, split tokens_b to fit within the maximum length """
    if model_type == 'roberta':
        double_line_token, newline_token = tokenizer.tokenize("a\n\n\n")[1:]
    elif model_type == 'xlm':
        double_line_token, newline_token = None, tokenizer.tokenize("a\n")[1]

    tokens_b_split = [] # stores tuple of (list of tokens, line_start)
    new_line_lists = [] # stores tuple of (number of tokens, index, num_of_newline)
    # logger.info("Code: %s", tokenizer.convert_tokens_to_string(tokens_b))

    prev_cutoff = 0
    b_max_length = max_length - len(tokens_a)

    for i, token in enumerate(tokens_b):
        if token == newline_token:
            new_line_lists.append((i-prev_cutoff+1, i, 1))
            prev_cutoff = i+1
        if token == double_line_token:
            new_line_lists.append((i-prev_cutoff+1, i, 2))
            prev_cutoff = i+1

    prev_index = -1
    cumul_num_tokens = 0
    prev_num_line = 0
    cumul_num_lines = 0
    for i, tup in enumerate(new_line_lists):
        num_tokens, index, num_lines = tup
        # logger.info("Tuple: %d, %d, %d" % tup)
        if cumul_num_tokens + num_tokens > b_max_length: # can't include current line
            if cumul_num_tokens > b_max_length:
                raise Exception("Failed to add line!")
            else:
                index_before = new_line_lists[i-1][1]
                tokens_b_split.append((tokens_b[prev_index+1:index_before+1], prev_num_line))
                # logger.info("Overshoot: %d / %d, append: %d, num_line: %d" % (cumul_num_tokens, b_max_length, len(tokens_b[prev_index+1:index_before+1]), prev_num_line))
                prev_index = index_before
                prev_num_line = cumul_num_lines
                cumul_num_tokens = num_tokens
                cumul_num_lines += num_lines

        else:
            cumul_num_lines += num_lines
            cumul_num_tokens += num_tokens
    
    if prev_index != len(tokens_b):
        if len(tokens_b[prev_index+1:]) > b_max_length:
            num_tokens, index, num_lines = new_line_lists[-1]
            # logger.info("Last line append: %d, prev_num_line %d" % (len(tokens_b[prev_index+1:index+1]), prev_num_line))
            tokens_b_split.append((tokens_b[prev_index+1:index+1], prev_num_line))
            # logger.info("remaining: %d, prev_num_line %d" % (len(tokens_b[index+1:]), cumul_num_lines+num_lines))
            tokens_b_split.append((tokens_b[index+1:], cumul_num_lines+num_lines))

        else:    
            tokens_b_split.append((tokens_b[prev_index+1:], prev_num_line))
            # logger.info("Overshoot end: %d / %d, append: %d" % (cumul_num_tokens, b_max_length, len(tokens_b[prev_index+1:])))

    return tokens_b_split
    

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1_macro = f1_score(y_true=labels, y_pred=preds, average='macro')
    f1_weighted = f1_score(y_true=labels, y_pred=preds, average='weighted')
    conf_matrix = confusion_matrix(labels, preds)
    return {
        "acc": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "acc_and_f1_macro": (acc + f1_macro) / 2,
        "acc_and_f1_weighted": (acc + f1_weighted) / 2,
        "confusion_matrix": conf_matrix
    }

def convert_token_index_to_lines(tokenizer, new_line_tokens, token_id, pred_start, pred_end, pred_prob, token_b_start, token_b_end, line_start, curr_index):
    logging = False
    d_line_token_id, s_line_token_id = new_line_tokens
    if logging: logger.info("new line tokens: %s" % str(new_line_tokens))
    
    non_null = False

    lines = []
    if logging:
        logger.info("len(token_id): %d" % len(token_id))
        logger.info("Question: %s" % tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(token_id[0][1:token_b_start[0]-1])))
    for i in range(len(token_id)): # iterate through all partials of an example
        top_k_lines = []
        for j in range(len(pred_start[0])):
            if logging:
                logger.info("Pred_start: %d, Pred_end: %d" % (pred_start[i][j], pred_end[i][j]))
                logger.info("token_b_start: %d, token_b_end: %d, line_start: %d" % (token_b_start[i], token_b_end[i], line_start[i]))
                logger.info("token at start: %s, token at end: %s" % (str(tokenizer.convert_ids_to_tokens([token_id[i][token_b_start[i]]])),  str(tokenizer.convert_ids_to_tokens([token_id[i][token_b_end[i]]]))))
                logger.info("token_id[i]: %s, pred_start[i][j]: %s\n" % (token_id[i], pred_start[i][j]))
                logger.info("token at pred_start: %s, token at pred_end: %s\n" % (str(tokenizer.convert_ids_to_tokens([token_id[i][int(pred_start[i][j])]])), \
                    str(tokenizer.convert_ids_to_tokens([token_id[i][int(pred_end[i][j])]]))))
                logger.info("pred_prob: %s" % (pred_prob))
                # logger.info("all tokens: %s" % str(token_id[i]))
                # logger.info("Code:\n%s" % tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(token_id[i])))

            # if pred_start[i] == 0 or pred_end[i] == 0 or (not ):
            if not (token_b_start[i] <= pred_start[i][j] <= token_b_end[i] and token_b_start[i] <= pred_end[i][j] <= token_b_end[i]) or pred_end[i][j] < pred_start[i][j]:
                top_k_lines.append((None, None, None, None, pred_prob[i][j]))
                continue
            
            non_null = True
            
            curr_line = line_start[i]
            l_b, l_e = None, None
            c_pred_start, c_pred_end = int(pred_start[i][j]) - token_b_start[i], int(pred_end[i][j]) - token_b_start[i]
            new_line_lists = []
            end_new_line = None
            # logger.info("token_b_start[i]: %d, token_b_end[i]+1: %d" % (token_b_start[i], token_b_end[i]+1))
            for k, tok in enumerate(token_id[i][token_b_start[i]:token_b_end[i]+1]): # iterate through tokens for one partial
                if tok == s_line_token_id:
                    if l_b == None and k >= c_pred_start:
                        l_b = curr_line
                    if l_e == None and k >= c_pred_end:
                        l_e = curr_line
                        end_new_line = 1
                    new_line_lists.append((curr_line, k, 1))
                    curr_line += 1
                elif tok == d_line_token_id:
                    if l_b == None and k >= c_pred_start:
                        l_b = curr_line
                    if l_e == None and k >= c_pred_end:
                        l_e = curr_line
                        end_new_line = 2
                    new_line_lists.append((curr_line, k, 2))
                    curr_line += 2
                elif k == token_b_end[i] - token_b_start[i]:
                    if l_b == None and k >= c_pred_start:
                        l_b = curr_line
                    if l_e == None and k >= c_pred_end:
                        l_e = curr_line
                        end_new_line = 0
            
            if logging: logger.info("new line lists: %s", str(new_line_lists))
            if logging: logger.info("len: %d, calculated: %d" % (len(token_id[i][token_b_start[i]:token_b_end[i]+1]) - 1, token_b_end[i] - token_b_start[i]))
            if logging: logger.info("l_b: %d, l_e: %d, curr_line: %d\n" % (l_b, l_e, curr_line))
            assert l_b != None and l_e != None
            
            # logger.info("i: %s, j: %s\n" % (i, j))
            top_k_lines.append((l_b, l_e, curr_line, end_new_line, pred_prob[i][j]))
        lines.append(top_k_lines)
        
    if non_null and logging:
        logger.info("lines for example %d: %s" % (curr_index, str(lines)))
    return lines

def calculate_F1_EM(selected_intervals, label):
    # logger.info("Selected_intervals: %s\n" % selected_intervals)
    if selected_intervals[0] == (None, None): # No selections
        if label[0] is None:
            return (1, 1)
        else:
            return (0, 0)
    elif label[0] is None:
        return (0, 0)
        
    label_set = set(range(label[0], label[1] + 1))
    num_lines_label = len(label_set)

    selection_set = set()
    for x in selected_intervals:
        curr_set = set(range(x[0], x[1]+1))
        selection_set = selection_set.union(curr_set)
    num_lines_selection = len(selection_set)

    tp = len(label_set.intersection(selection_set))
    if tp == 0:
        return (0, 0)
    fp = len(selection_set.difference(label_set))
    fn = len(label_set.difference(selection_set))

    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    F1 = 2*precision*recall / (precision+recall)
    EM = 1 if F1 == 1 else 0

    return F1, EM

def extract_top_k_combinations(all_lines, k):
    q = PriorityQueue()
    num_parts = len(all_lines)

    sums = np.empty([k for i in range(num_parts)])
    visited = np.zeros([k for i in range(num_parts)], dtype=bool)

    def increment_base_k(l, k):
        curr_index = -1
        l[curr_index] += 1
        while curr_index != -len(l) and l[curr_index] == k:
            l[curr_index] = 0
            curr_index -= 1
            l[curr_index] += 1
        return l

    line_indices = [0 for i in range(num_parts)]
    for i in range(k ** num_parts):
        curr_sum = 0
        # logger.info("line_indicess: %s\n" % (line_indices))
        for j, x in enumerate(line_indices):
            curr_sum -= all_lines[j][x][4]
        sums[tuple(line_indices)] = curr_sum
        increment_base_k(line_indices, k)

    # logger.info("Sums: %s" % (sums))

    return_list = []

    init_pos = [0 for i in range(num_parts)]
    q.put((sums[tuple(init_pos)], init_pos))
    num_added = k
    while num_added > 0:
        num_added -= 1
        x = q.get()
        return_list.append(x)
        _, curr_pos = x

        # Calculating the new positions
        for i in range(num_parts):
            # logger.info("curr_pos: %s\n" % (curr_pos))
            new_pos = curr_pos[:]
            new_pos[i] += 1
            if new_pos[i] < k and not visited[tuple(new_pos)]:
                q.put((sums[tuple(new_pos)], new_pos))
                visited[tuple(new_pos)] = True
    
    # return_list now contains top 5 combinations, with the second element indicating the 
    # index for the parts in each code-part

    # logger.info("return_list: %s\n" % (return_list))
    return return_list

def compute_metrics(task_name, tokenizer, token_ids, pred_starts, pred_ends, pred_probs, token_b_starts, token_b_ends, line_starts, spans_by_index, ex_labels, model_type):
    eval_by_type = ex_labels != []
    cumul_scores_by_type = [[0 for i in range(11)], [0 for i in range(11)], [0 for i in range(11)]]

    if model_type == 'roberta':
        new_line_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("a\n\n\n")[1:])
    elif model_type == 'xlm':
        new_line_tokens = None, tokenizer.convert_tokens_to_ids(tokenizer.tokenize("a\n")[1])
    
    # assert len(preds) == len(labels)
    cumul_F1, cumul_EM = 0, 0
    if task_name == "typeclassification":
        return acc_and_f1(preds, labels)
    elif task_name == "spanselection":
        selections = []
        candidates = []
        for i in range(len(token_ids)):
            # if i == 20: break
            # logger.info("######################## Entering example %d ########################" % i)
            
            all_lines = convert_token_index_to_lines(tokenizer, new_line_tokens, token_ids[i], pred_starts[i], pred_ends[i], pred_probs[i], token_b_starts[i],\
                                                 token_b_ends[i], line_starts[i], i)

            # Extract 5 best combinations of selections over the separated parts of a single question
            top_k_selections = extract_top_k_combinations(all_lines, k)
            # logger.info("top_5_selections: %s\n" % top_5_selections)
            top_k_final = []
            for top_k_index, curr_selection in enumerate(top_k_selections):
                top_indices = curr_selection[1]
                lines = [all_lines[j][part_index] for j,part_index in enumerate(top_indices)]

                selections_made = False
                for part_selection in lines:
                    if part_selection[0] != None:
                        selections_made = True
                        break

                # logger.info("selections_made: %s\n", (selections_made))

                curr_selection = [[]]
                if not selections_made: # all parts returned no selection
                    curr_selection[0].append((None, None))
                    top_k_final.append(curr_selection)
                    # labels = spans_by_index[i]
                    # scores = [calculate_F1_EM(curr_selection, x) for x in labels]
                    # best_score = scores[np.argmax([x[0] for x in scores])]

                    # curr_selection.extend([best_score[0], best_score[1]])
                else: # some selections made
                    curr_begin, curr_end, curr_line_end, curr_end_new_line, prob = lines.pop(0)
                    while len(lines) > 0:
                        curr_part = lines.pop(0)
                        if curr_begin != None and curr_part[0] == curr_line_end and curr_end == curr_line_end - curr_end_new_line:
                            curr_end = curr_part[1]
                            curr_line_end = curr_part[2]
                        else:
                            if curr_begin != None:
                                curr_selection[0].append((curr_begin, curr_end))
                            curr_begin, curr_end, curr_line_end, curr_end_new_line, prob = curr_part
                    if curr_begin != None:
                        curr_selection[0].append((curr_begin, curr_end))
                    top_k_final.append(curr_selection)

                    # logger.info("converted lines: %s" % str(curr_selection))

                labels = spans_by_index[i]
                scores = [calculate_F1_EM(curr_selection[0], x) for x in labels]

                best_score = scores[np.argmax([x[0] for x in scores])]
                curr_selection.extend([best_score[0], best_score[1]])

                # logger.info("curr_selection: %s\n" % curr_selection)
                # logger.info("top_5_final: %s\n" % top_5_final)

            # logger.info("top_5_final: %s\n" % top_5_final)
            top_combination = top_k_final[np.argmax([x[1] for x in top_k_final])]
            selections.append((top_k_final, np.argmax([x[1] for x in top_k_final])))

            best_score = top_combination[1:]

            cumul_F1 += best_score[0]
            cumul_EM += best_score[1]

            if best_score[0] >= 0:
                candidates.append((i, curr_selection))

            if eval_by_type:
                label_list = ["code_understanding", "logical", "error", "usage", "algorithm", "task", "comparison", "reasoning", "code_explain", "variable", "guiding"]
                cumul_scores_by_type[0][label_list.index(ex_labels[i])] += best_score[0]
                cumul_scores_by_type[1][label_list.index(ex_labels[i])] += best_score[1]
                cumul_scores_by_type[2][label_list.index(ex_labels[i])] += 1
            # logging.info("labels: %s" % str(labels))
            # logging.info("curr_selection: %s" % str(curr_selection))
            # logging.info("scores: %s" % str(scores))
            # logging.info("best score: %s\n" % str(best_score))

        return_dict = {
            "F1": cumul_F1 / len(token_ids),
            "EM": cumul_EM / len(token_ids)
        }
        if eval_by_type:
            logger.info("cumul_scores_by_type: %s" % str(cumul_scores_by_type))
            for i in range(11):
                if cumul_scores_by_type[2][i] != 0:
                    cumul_scores_by_type[0][i] /= cumul_scores_by_type[2][i]
                    cumul_scores_by_type[1][i] /= cumul_scores_by_type[2][i]
            return_dict['F1_type'] = cumul_scores_by_type[0]
            return_dict['EM_type'] = cumul_scores_by_type[1]

        # logging.info("Candidates: %s" % (candidates) )
        return return_dict, selections

    else:
        raise KeyError(task_name)


processors = {
    "spanselection": SpanSelectionProcessor,
}

output_modes = {
    "spanselection": "questionanswering",
}

GLUE_TASKS_NUM_LABELS = {
    "spanselection": 2,
}
