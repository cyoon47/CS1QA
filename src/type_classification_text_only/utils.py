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

csv.field_size_limit(sys.maxsize)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, labNo, taskNo, questioner, question, code, label=None):
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
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, example_id=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.example_id = example_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

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


class TypeclassificationProcessor(DataProcessor):
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

    def get_test_examples(self, data_dir, test_file):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, test_file)))
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, test_file)), "test")

    def get_labels(self):
        """See base class."""
        return ["code_understanding", "logical", "error", "usage", "algorithm", "task", "comparison", "reasoning", "code_explain", "variable", "guiding"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        # guid, labNo, taskNo, questioner, question, code, label
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            labNo = line['labNo']
            taskNo = line['taskNo']
            questioner = line['questioner']
            question = line['question']
            code = ''

            if (set_type == 'test'):
                label = line['questionType']
            else:
                label = line['questionType']
            examples.append(
                InputExample(guid=guid, labNo=labNo, taskNo=taskNo, questioner=questioner,\
                            question=question, code=code, label=label))
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
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        #labNo, taskNo, questioner, question, code, label
        text_a = ' '.join([str(example.labNo), str(example.taskNo), example.questioner, example.question])
        tokens_a = tokenizer.tokenize(text_a)#[:50]

        tokens_b = None
        if example.code:
            tokens_b = tokenizer.tokenize(example.code)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            
            # _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            tokens_b_split = _split_long_tokens(tokenizer, tokens_a, tokens_b, max_seq_length - 3, model_type)
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
        tokens = tokens_a + [sep_token]
        # logger.info("tokens_a length: %d" % len(tokens_a))
        segment_ids = [sequence_a_segment_id] * len(tokens)
        all_tokens = []
        all_segments = []
        
        curr_token = tokens[:]
        curr_segment = segment_ids[:]

        if cls_token_at_end:
            curr_token = curr_token + [cls_token]
            curr_segment = curr_segment + [cls_token_segment_id]
        else:
            curr_token = [cls_token] + curr_token
            curr_segment = [cls_token_segment_id] + curr_segment
        

        input_ids = tokenizer.convert_tokens_to_ids(curr_token)
        # logger.info("input_ids length: %d" % len(input_ids))
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

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in curr_token]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in curr_segment]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))
        
        features.append(
            InputFeatures(input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=curr_segment,
                        label_id=label_id, 
                        example_id=ex_index))
    return features


def _split_long_tokens(tokenizer, tokens_a, tokens_b, max_length, model_type):
    """When total token length is greater than the maximum length, split tokens_b to fit within the maximum length """
    if model_type == 'roberta':
         double_line_token, newline_token = tokenizer.tokenize("a\n\n\n")[1:]
    elif model_type == 'xlm':
        double_line_token, newline_token = None, tokenizer.tokenize("a\n")[1]

    tokens_b_split = [] # stores tuple of (list of tokens, line_start)
    new_line_lists = [] # stores tuple of (number of tokens, index, num_of_newline)

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
            tokens_b_split.append((tokens_b[prev_index+1:], cumul_num_lines))
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
    np_conf = np.array(conf_matrix)
    class_p = []
    class_r = []
    class_f1 = []
    for i in range(len(np_conf)):
        recall = np_conf[i,i] / sum(np_conf[i,:])
        precision = np_conf[i,i] / sum(np_conf[:,i])
        f1 = recall*precision / (recall+precision) * 2

        class_p.append(precision)
        class_r.append(recall)
        class_f1.append(f1)

    return {
        "acc": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "acc_and_f1_macro": (acc + f1_macro) / 2,
        "acc_and_f1_weighted": (acc + f1_weighted) / 2,
        "confusion_matrix": conf_matrix,
        "class_p": class_p,
        "class_r": class_r,
        "class_f1": class_f1
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "typeclassification":
        return acc_and_f1(preds, labels)
    else:
        raise KeyError(task_name)


processors = {
    "typeclassification": TypeclassificationProcessor,
}

output_modes = {
    "typeclassification": "classification",
}

GLUE_TASKS_NUM_LABELS = {
    "typeclassification": 2,
}
