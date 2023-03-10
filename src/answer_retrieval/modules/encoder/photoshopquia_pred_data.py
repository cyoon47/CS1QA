#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
from dataclasses import dataclass, field
import logging
import random
import json
import jsonlines
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
from tqdm import tqdm
from typing import Optional

from .tokenizer import get_tokenizer

logger = logging.getLogger(__name__)

@dataclass
class PhotoshopQuiAPredDataArguments:
    max_length: Optional[int] = field(default=128)
    eval_file: Optional[str] = field(default=None)
    bm25_file: Optional[str] = field(default=None)
    corpus_file: Optional[str] = field(default=None)

class PhotoshopQuiACrossEncoderPredDataset(Dataset):
    def __init__(self, input_file, tokenizer, config):
        self.config = config
        self.tokenizer = tokenizer
        self.max_length \
            = self.config.max_length
        self.id2doc = self.read_corpus(self.config.corpus_file)
        self.qid2bm25_docs = self.read_bm25_file(self.config.bm25_file)
        self.data = \
            self.get_data(input_file)
    
    def read_corpus(self, corpus_file):
        id2doc = {}
        with jsonlines.open(corpus_file, "r") as reader:
            for r in tqdm(reader, desc="Reading the corpus"):
                id2doc[r["id"]] = "{}".format(r["text"])
        return id2doc
    
    def read_bm25_file(self, bm25_file):
        qid2bm25_docs = {}
        with jsonlines.open(bm25_file, "r") as reader:
            for r in reader:
                bm25_docs = r["bm25_results"]
                bm25_docs.sort(key=lambda x: x["bm25_score"], reverse=True)
                bm25_docs = [str(elem["doc_id"]) for elem in bm25_docs]
                qid2bm25_docs[r["qid"]] = bm25_docs
        return qid2bm25_docs
    
    def get_data(self, fname):
        data = []
        with jsonlines.open(fname, "r") as reader:
            data = [r for r in reader]

        target_data = []
        for d in tqdm(data, desc="Preprocessing the qas dataset"):
            qid = d["id"]
            positive = str(d["doc_id"])
            bm25 = self.qid2bm25_docs[qid][:50]
            if positive not in bm25:
                bm25[-1] = positive
            pos_ind = -1
            for i, doc_id in enumerate(bm25):
                if doc_id == positive:
                    pos_ind = i
            target_data.append({
                "gt": positive,
                "gt_ind": pos_ind,
                "cands": bm25,
                "question": \
                    "{} {}".format( \
                        d["title"], d["text"] \
                    )
            })
        return target_data
    
    def tokenize(self, inps):
        tokenized = self.tokenizer.batch_encode_plus( \
            inps, \
            padding="max_length", \
            max_length=self.max_length, \
            truncation=True, \
            return_tensors='pt' \
        )
        return tokenized

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        question = example["question"]
        cands = example["cands"]
        gt_ind = example["gt_ind"]
        
        features = self.tokenize( \
            [question] + [self.id2doc[doc_id] for doc_id in cands] \
        )

        features["n"] = len(cands) + 1
        features["gt_ind"] = gt_ind
        return features

def photoshopquia_pred_data_collator(samples):
    if len(samples) == 0:
        return {}
    
    bsize = len(samples) 
    
    input_ids = []
    token_type_ids = []
    attention_mask = []
    gt_inds = []
    ns = []
    for inp in samples:
        input_ids.append( \
            inp["input_ids"] \
        )
        token_type_ids.append( \
            inp["token_type_ids"] if "token_type_ids" in inp else None\
        )
        attention_mask.append( \
            inp["attention_mask"] \
        )
        gt_inds.append(inp["gt_ind"])
        ns.append(inp["n"])
    
    input_ids = torch.cat(input_ids, dim=0)
    if token_type_ids[0] != None:
        token_type_ids = torch.cat( \
            token_type_ids, dim=0 \
        )
    else:
        token_type_ids = None
    attention_mask = torch.cat( \
        attention_mask, dim=0 \
    )
    gt_inds = torch.tensor(gt_inds)
    ns = torch.tensor(ns)
    
    new_batch = {
        "n_samples": bsize,
        "ns": ns,
        "gt_inds": gt_inds,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    if token_type_ids != None:
        new_batch["token_type_ids"] = token_type_ids
    return new_batch 
