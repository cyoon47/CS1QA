import os

from accelerate import Accelerator
import apex
import collections
from dataclasses import dataclass, field
import logging
import math
import numpy as np
import json
import jsonlines
import random
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import ( 
    BertTokenizer, 
    BertModel, 
    BertPreTrainedModel, 
    RobertaPreTrainedModel, 
    RobertaModel, 
    AdamW, 
    get_scheduler,
    get_linear_schedule_with_warmup
)
from typing import Optional

from .tokenizer import get_tokenizer
from .train_data import CrossEncoderDataset, data_collator
from .photoshopquia_train_data import PhotoshopQuiACrossEncoderDataset, photoshopquia_data_collator
from .pred_data import CrossEncoderPredDataset, pred_data_collator
from .photoshopquia_pred_data import PhotoshopQuiACrossEncoderPredDataset, photoshopquia_pred_data_collator

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    init_checkpoint: Optional[str] = field(default=None)
    checkpoint_save_dir: Optional[str] = field(default=None)
    dataset: Optional[str] = field(default="techqa")
    per_device_train_batch_size: Optional[int] = field(default=16)
    per_device_eval_batch_size: Optional[int] = field(default=16)
    gradient_accumulation_steps: Optional[int] = field(default=16)
    eval_steps: Optional[int] = field(default=16)
    n_epochs: Optional[int] = field(default=3)
    weight_decay: Optional[float] = field(default=0.01)
    learning_rate: Optional[float] = field(default=2e-5)
    warmup_ratio: Optional[float] = field(default=0.1)
    backbone_model: Optional[str] = field(default="bert")

@dataclass
class PredEncoderArguments:
    dataset: Optional[str] = field(default="techqa")
    init_checkpoint: Optional[str] = field(default=None)
    per_device_eval_batch_size: Optional[int] = field(default=16)
    eval_steps: Optional[int] = field(default=16)
    backbone_model: Optional[str] = field(default="bert")

class BERTForTextSim(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.roberta = RobertaModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.layer1 = nn.Linear(config.hidden_size, config.hidden_size)

        self.init_weights() 

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.layer1(pooled_output)
        
        return logits

class Trainer:
    def __init__(self, model_config, data_config):
        self.model_config = model_config
        self.data_config = data_config
        
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        self.n_gpus = torch.cuda.device_count()
        print("N GPUs: {:d}".format(self.n_gpus))

        self.tokenizer =  \
            get_tokenizer( \
                self.model_config.init_checkpoint, \
                self.model_config.backbone_model \
            )
        self.model = self.get_model()
        self.model = self.model.to(self.device)

        self.train_dataloader = None
        self.eval_dataloader = None
        assert self.data_config.train_file != None \
            and self.data_config.eval_file != None
        self.train_dataset, \
            self.train_dataloader = \
                self.get_data( \
                    self.data_config.train_file, \
                    is_train=True \
                )
        self.eval_dataset, \
            self.eval_dataloader = \
                self.get_data( \
                    self.data_config.eval_file, \
                    is_train=False \
                )
        
        self.optimizer, \
            self.lr_scheduler, \
            self.total_steps = \
            self.get_optimizer()
        
        apex.amp.register_half_function(torch, 'einsum')
        self.model, self.optimizer \
            = apex.amp.initialize(
                self.model, self.optimizer, \
                opt_level="O1")
        
        if self.n_gpus > 1:
            self.model \
                = torch.nn.DataParallel( \
                    self.model)
        self.model.eval()
    
    def get_model(self):
        model = BERTForTextSim.from_pretrained(
            self.model_config.init_checkpoint,
        )
        return model
    
    def get_optimizer(self):
        assert self.train_dataloader != None
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [ \
                    p for n, p in self.model.named_parameters() \
                        if not any(nd in n for nd in no_decay) \
                ],
                "weight_decay": self.model_config.weight_decay,
            },
            {
                "params": [ \
                    p for n, p in self.model.named_parameters() \
                        if any(nd in n for nd in no_decay) \
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW( \
            optimizer_grouped_parameters, \
            lr=self.model_config.learning_rate \
        )
       
        n_steps_per_epoch = \
            math.ceil( \
                len(self.train_dataloader)\
                  / self.model_config.gradient_accumulation_steps \
            )
        train_steps = self.model_config.n_epochs \
                        * n_steps_per_epoch
        n_warmup_steps = int( \
            train_steps * self.model_config.warmup_ratio \
        )
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=n_warmup_steps,
            num_training_steps=train_steps,
        )
        return (optimizer, lr_scheduler, train_steps)
    
    def save_model(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        model_to_save = \
            self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print("\n")
        print("Saving model checkpoint to {:s}".format(output_dir))
        return 0

    def get_data(self, input_file, is_train):
        dataset2class = {
            "photoshopquia": PhotoshopQuiACrossEncoderDataset,
            "techqa": CrossEncoderDataset
        }
        dataset2data_collator = {
            "photoshopquia": photoshopquia_data_collator,
            "techqa": data_collator
        }
        target_class = dataset2class[self.model_config.dataset]
        target_data_collator = dataset2data_collator[self.model_config.dataset]
        dataset_instance = target_class( \
            input_file, \
            self.tokenizer, \
            self.data_config \
        )
        dataloader_instance = DataLoader(
            dataset_instance,
            shuffle=is_train,
            collate_fn=target_data_collator,
            pin_memory=True,
            batch_size= \
                self.model_config.per_device_train_batch_size * self.n_gpus \
                    if is_train \
                    else self.model_config.per_device_eval_batch_size * self.n_gpus, \
            num_workers=self.n_gpus
        )
        return (dataset_instance, dataloader_instance)
    
    def get_loss(self, logits, n):
        total_n, _ = logits.size()
        q_vecs, doc_vecs = \
            torch.split( \
                logits, \
                [n, total_n - n], \
                dim=0 \
            )
        sim = torch.matmul(q_vecs, doc_vecs.transpose(0, 1))
        pos_indices = torch.arange(n).to(sim.get_device())
        
        loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(sim, pos_indices)
        return loss

    def train_model(self):
        total_batch_size = \
            self.model_config \
                .per_device_train_batch_size \
            * self.n_gpus \
            * self.model_config.gradient_accumulation_steps
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num Epochs = {self.model_config.n_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.model_config.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.model_config.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.total_steps}")
        
        last_checkpoint_dir = "{}_{}".format(self.model_config.checkpoint_save_dir.rstrip("/"), "last_checkpoint")
        total_train_loss = 0.0
        best_val_acc = float("-inf")
        global_steps = 0
        self.model.zero_grad()
        for i in range(0, self.model_config.n_epochs):
            self.model.train()
            progress_bar = tqdm( \
                self.train_dataloader, \
                desc="Training ({:d}'th iter / {:d})".format(i+1, self.model_config.n_epochs, 0.0, 0.0) \
            )
            for step, batch in enumerate(progress_bar):
                self.model.train()
                target_inputs = [ \
                    "input_ids", "token_type_ids", \
                    "attention_mask", \
                ]
                new_batch = {k: batch[k].to(self.device) for k in target_inputs if k in batch}
                logits = self.model(**new_batch)
                loss = self.get_loss(logits, batch["n_samples"])
                loss = loss / self.model_config.gradient_accumulation_steps
                
                with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                
                if step % self.model_config.gradient_accumulation_steps \
                        == self.model_config.gradient_accumulation_steps - 1 \
                    or step == len(self.train_dataloader) - 1:
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    global_steps += 1
                total_train_loss += loss.item()
                
                if global_steps % self.model_config.eval_steps == 1:
                    avg_train_loss = \
                        total_train_loss \
                            / self.model_config.eval_steps
                    val_acc = self.eval_model()
                    if val_acc > best_val_acc:
                        print("saving checkpoint")
                        self.save_model(self.model_config.checkpoint_save_dir)
                        best_val_acc = val_acc
                    self.save_model(last_checkpoint_dir)
                    desc_template = "Training ({:d}'th iter / {:d})|loss: {:.03f}, Val: {:.03f}"
                    print('\n')
                    print(desc_template.format( \
                            i+1, \
                            self.model_config.n_epochs, \
                            avg_train_loss, \
                            val_acc \
                        )
                    )
                    total_train_loss = 0.0
        val_acc = self.eval_model()
        if val_acc > best_val_acc:
            self.save_model(self.model_config.checkpoint_save_dir)
            best_val_acc = val_acc
        self.save_model(last_checkpoint_dir)
        return 0
    
    def get_acc(self, logits, n):
        total_n, _ = logits.size()
        q_vecs, doc_vecs = \
            torch.split( \
                logits, \
                [n, total_n - n], \
                dim=0 \
            )
        sim = torch.matmul(q_vecs, doc_vecs.transpose(0, 1)).numpy()
        return (np.argmax(sim, axis=1) == np.arange(n)) * 1.0

    def eval_model(self):
        target_inputs = [ \
            "input_ids", "token_type_ids", \
            "attention_mask" \
        ]
        self.model.eval()
        scores = []
        for batch in tqdm(self.eval_dataloader, desc="Eval"):
            with torch.no_grad():
                new_batch = {k: batch[k].to(self.device) for k in target_inputs if k in batch}
                results = self.model(**new_batch)
                preds = results.detach().cpu()
                scores += self.get_acc(preds, batch["n_samples"]).tolist()

        acc = np.mean(scores) * 100.0
        print("Eval: {:.04f}".format(acc))
        return acc

class Predictor:
    def __init__(self, model_config, data_config):
        self.model_config = model_config
        self.data_config = data_config
        
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        self.n_gpus = torch.cuda.device_count()
        print("N GPUs: {:d}".format(self.n_gpus))

        self.tokenizer =  \
            get_tokenizer( \
                self.model_config.init_checkpoint, \
                self.model_config.backbone_model \
            )
        self.model = self.get_model()
        self.model = self.model.to(self.device)

        self.eval_dataloader = None
        assert self.data_config.eval_file != None
        self.eval_dataset, \
            self.eval_dataloader = \
                self.get_data( \
                    self.data_config.eval_file, \
                )
        
        apex.amp.register_half_function(torch, 'einsum')
        self.model = apex.amp.initialize(
            self.model, \
            opt_level="O1" \
        )
        
        if self.n_gpus > 1:
            self.model \
                = torch.nn.DataParallel( \
                    self.model)
        self.model.eval()
    
    def get_model(self):
        model = BERTForTextSim.from_pretrained(
            self.model_config.init_checkpoint,
        )
        return model
    
    def get_data(self, input_file):
        dataset2class = {
            "photoshopquia": PhotoshopQuiACrossEncoderPredDataset,
            "techqa": CrossEncoderPredDataset
        }
        dataset2data_collator = {
            "photoshopquia": photoshopquia_pred_data_collator,
            "techqa": pred_data_collator
        }
        target_class = dataset2class[self.model_config.dataset]
        target_data_collator = dataset2data_collator[self.model_config.dataset]
        
        dataset_instance = target_class( \
            input_file, \
            self.tokenizer, \
            self.data_config \
        )
        dataloader_instance = DataLoader(
            dataset_instance,
            shuffle=False,
            collate_fn=target_data_collator,
            pin_memory=True,
            batch_size= \
                self.model_config.per_device_eval_batch_size * self.n_gpus, \
            num_workers=self.n_gpus
        )
        return (dataset_instance, dataloader_instance)
    
    def get_top1(self, preds, gts):
        top1 = []
        top5 = []
        mrr = []
        upperbound = []
        for pred, gt in zip(preds, gts):
            if gt.numpy().tolist() == -1:
                top1.append(0.0)
                top5.append(0.0)
                mrr.append(0.0)
                upperbound.append(0.0)
                continue
            upperbound.append(1.0)
            
            scores = np.dot(pred[0], pred[1:].T)

            topk_inds = np.argsort(-1.0 * scores).tolist()
            gt_ind = gt.numpy().tolist()
            
            if topk_inds[0] == gt_ind:
                top1.append(1.0)
            else:
                top1.append(0.0)

            if gt_ind in topk_inds[:5]:
                top5.append(1.0)
            else:
                top5.append(0.0)
            
            mrr.append(1.0 / (1.0 + topk_inds.index(gt_ind)))

        return (top1, top5, mrr, upperbound)

    def eval_model(self):
        target_inputs = [ \
            "input_ids", "token_type_ids", \
            "attention_mask" \
        ]
        self.model.eval()
        top1 = []
        top5 = []
        mrr = []
        upper = []
        for batch in tqdm(self.eval_dataloader, desc="Eval"):
            with torch.no_grad():
                new_batch = {k: batch[k].to(self.device) for k in target_inputs if k in batch}
                results = self.model(**new_batch)
                preds = results.detach().cpu()
                gts = batch["gt_inds"].detach().cpu()
                
                ns = batch["ns"].cpu().numpy().tolist()
                pred_list = torch.split( \
                    preds, ns, dim=0 \
                )
                t1, t5, m, u = self.get_top1(pred_list, gts)
                top1 += t1
                top5 += t5
                mrr += m
                upper += u
        top1 = np.mean(top1) * 100.0
        top5 = np.mean(top5) * 100.0
        mrr = np.mean(mrr) * 100.0
        print("Eval: ({:.04f}, {:.04f}, {:.04f}) / {:.04f}".format(top1, top5, mrr, np.mean(upper) * 100.0))
        return top1

