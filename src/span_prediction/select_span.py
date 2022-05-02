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
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet)."""
import argparse
import glob
import logging
import os
import random
import jsonlines
import pickle

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from queue import PriorityQueue

from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup, AdamW,
                          RobertaConfig,
                          RobertaForQuestionAnswering,
                          RobertaTokenizer, XLMRobertaConfig, XLMRobertaForQuestionAnswering, XLMRobertaTokenizer,
                          LongformerTokenizer, LongformerConfig, LongformerForQuestionAnswering)

from utils import (compute_metrics, convert_examples_to_features,
                        output_modes, processors)

from span_prediction.arguments import add_arguments

logger = logging.getLogger(__name__)

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaForQuestionAnswering, RobertaTokenizer),
                 'xlm': (XLMRobertaConfig, XLMRobertaForQuestionAnswering, XLMRobertaTokenizer),
                 'longformer': (LongformerConfig, LongformerForQuestionAnswering, LongformerTokenizer)}

def _read_jsonl(input_file):
    with jsonlines.open(input_file, 'r') as f:
        lines = [l for l in f]
    return lines

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def train(args, train_dataset, model, tokenizer, optimizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, t_total)

    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last))

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = args.start_step
    tr_loss, logging_loss = 0.0, 0.0
    best_acc = 0.0
    model.zero_grad()
    train_iterator = trange(args.start_epoch, int(args.num_train_epochs), desc="Epoch",
                            disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    model.train()
    for idx, _ in enumerate(train_iterator):
        tr_loss = 0.0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):

            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                      # XLM don't use segment_ids
                      'start_positions': batch[3],
                      'end_positions': batch[4]}
            ouputs = model(**inputs)
            loss = ouputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                try:
                    from apex import amp
                except ImportError:
                    raise ImportError(
                        "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer, checkpoint=str(global_step))[0]
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                            logger.info('loss %s', str(tr_loss - logging_loss))
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss
            if args.max_steps > 0 and global_step > args.max_steps:
                # epoch_iterator.close()
                break

        if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
            results, attentions, global_attentions = evaluate(args, model, tokenizer, checkpoint=str(args.start_epoch + idx))

            last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
            if not os.path.exists(last_output_dir):
                os.makedirs(last_output_dir)
            model_to_save = model.module if hasattr(model,
                                                    'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(last_output_dir)
            logger.info("Saving model checkpoint to %s", last_output_dir)
            idx_file = os.path.join(last_output_dir, 'idx_file.txt')
            with open(idx_file, 'w', encoding='utf-8') as idxf:
                idxf.write(str(args.start_epoch + idx) + '\n')

            torch.save(optimizer.state_dict(), os.path.join(last_output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(last_output_dir, "scheduler.pt"))
            logger.info("Saving optimizer and scheduler states to %s", last_output_dir)

            step_file = os.path.join(last_output_dir, 'step_file.txt')
            with open(step_file, 'w', encoding='utf-8') as stepf:
                stepf.write(str(global_step) + '\n')

            if (results['F1'] > best_acc):
                best_acc = results['F1']
                output_dir = os.path.join(args.output_dir, 'checkpoint-best')
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model.module if hasattr(model,
                                                        'module') else model  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, 'training_{}.bin'.format(idx)))
                logger.info("Saving model checkpoint to %s", output_dir)

                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                logger.info("Saving optimizer and scheduler states to %s", output_dir)

                # with open(os.path.join(output_dir, "attention.pkl"), "wb") as attention_file:
                #     pickle.dump(attentions, attention_file)
                # with open(os.path.join(output_dir, "global_attention.pkl"), "wb") as global_attention_file:
                #     pickle.dump(global_attentions, global_attention_file)
                # logger.info("Saving attention and global attentions to %s", output_dir)

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step

def file_to_spans_list(file_path, return_type, test_example_begin, test_example_end):
    spans = []
    labels = []
    lines = _read_jsonl(file_path)

    for line in lines:
        if return_type: labels.append(line['questionType'])
        spans.append([])
        for l in lines:
            if line['labNo'] == l['labNo'] and line['taskNo'] == l['taskNo'] and line['questioner'] == l['questioner'] and\
                line['question'] == l['question'] and line['code'] == l['code']:
                spans[-1].append((l['startLine'], l['endLine']))
        if len(spans[-1]) == 4:
            pass
            # logger.info("duplicate 4: %s" % line)
    
    logger.info("Maximum duplicates: %d" % max(map(len, spans)))
    logger.info("Minimum duplicates: %d" % min(map(len, spans)))
    return spans[test_example_begin:test_example_end], labels[test_example_begin:test_example_end]

def get_top_k_indices(start_logits, end_logits, k):
    q = PriorityQueue()
    arg_start = np.flip(np.argsort(start_logits))
    arg_end = np.flip(np.argsort(end_logits))

    sorted_start = sorted(start_logits, reverse=True)
    sorted_end = sorted(end_logits, reverse=True)
    m, n = len(start_logits), len(end_logits)
    sums = [[] for i in range(k)]
    visited = [[] for i in range(k)]

    top_indices = []

    # logger.info("M and N: %d, %d" % (m,n))
    
    for i in range(k):
        for j in range(k):
            sums[i].append(-sorted_start[i] - sorted_end[j])
            visited[i].append(False)

    q.put((sums[0][0], 0, 0))
    while k > 0:
        k -= 1
        x = q.get()
        top_indices.append(x)
        s, i, j = x
        if i < k-1 and not visited[i+1][j]:
            q.put((sums[i+1][j], i+1, j))
            visited[i+1][j] = True
        if j < k-1 and not visited[i][j+1]:
            q.put((sums[i][j+1], i, j+1))
            visited[i][j+1] = True

    indices = [(arg_start[i], arg_end[j], -prob) for prob, i, j in top_indices]

    return indices


def evaluate(args, model, tokenizer, checkpoint=None, prefix="", mode='dev'):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    input_file_path = None
    if mode == 'dev':
        input_file_path = os.path.join(args.data_dir, args.dev_file)
    elif mode == 'test':
        input_file_path = os.path.join(args.data_dir, args.test_file)
    else:
        raise KeyError(mode)

    results = {}
    all_attentions, all_global_attentions = [], []
    
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        if (mode == 'dev'):
            eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, ttype='dev')
        elif (mode == 'test'):
            eval_dataset, instances = load_and_cache_examples(args, eval_task, tokenizer, ttype='test')
            output_test_file = args.test_result_dir
            output_dir = os.path.dirname(output_test_file)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        token_ids, pred_start, pred_end = None, None, None
        example_ids, line_starts, token_b_starts, token_b_ends = None, None, None, None
        # out_start_pos, out_end_pos = None, None

        if (mode == 'test'):
            attention_file = open(os.path.join(output_dir, "attention.pkl"), "wb")
            global_attention_file = open(os.path.join(output_dir, "global_attention.pkl"), "wb")

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                          # XLM don't use segment_ids
                          'start_positions': batch[3],
                          'end_positions': batch[4],
                          'output_attentions': True}
                
                l_starts = batch[5]
                b_starts = batch[6]
                b_ends = batch[7]
                ex_ids = batch[8]

                outputs = model(**inputs)
                tmp_eval_loss, start_logits, end_logits, attention, global_attentions = outputs[:5]
                
                # logger.info("Shape of attention and global_attentions: %s, %s\n" % (attention[0].shape, global_attentions[0].shape))
                # logger.info("Number of layers: %d\n" % (len(attention)))
                # # logger.info("start_logits shape: %s" % str(start_logits.size()))
                # logger.info("length of the 9th token: %d\n" % (len(attention[0][0][:][9])))
                # logger.info("Attention for 9th token: %s\n" % (attention[0][0][0][0]))
                if (mode == 'test'):
                    pickle.dump([x.detach().cpu().numpy() for x in attention], attention_file)
                    pickle.dump([x.detach().cpu().numpy() for x in global_attentions], global_attention_file)
                    # all_attentions.append([x.detach().cpu().numpy() for x in attention])
                    # all_global_attentions.append([x.detach().cpu().numpy() for x in global_attentions])

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if pred_start is None:
                pred_start = start_logits.detach().cpu().numpy()
                pred_end = end_logits.detach().cpu().numpy()
                # out_start_pos = inputs['start_positions'].detach().cpu().numpy()
                # out_end_pos = inputs['end_positions'].detach().cpu().numpy()
                example_ids = ex_ids.detach().cpu().numpy()
                token_ids = batch[0].detach().cpu().numpy()
                token_b_starts = b_starts.detach().cpu().numpy()
                token_b_ends = b_ends.detach().cpu().numpy()
                line_starts = l_starts.detach().cpu().numpy()
            else:
                pred_start = np.append(pred_start, start_logits.detach().cpu().numpy(), axis=0)
                pred_end = np.append(pred_end, end_logits.detach().cpu().numpy(), axis=0)
                # out_start_pos = np.append(out_start_pos, inputs['start_positions'].detach().cpu().numpy(), axis=0)
                # out_end_pos = np.append(out_end_pos, inputs['end_positions'].detach().cpu().numpy(), axis=0)
                example_ids = np.append(example_ids, ex_ids.detach().cpu().numpy(), axis=0)
                # logger.info("example_id: %s" % example_ids)
                token_ids = np.append(token_ids, batch[0].detach().cpu().numpy(), axis=0)
                token_b_starts = np.append(token_b_starts, b_starts.detach().cpu().numpy(), axis=0)
                token_b_ends = np.append(token_b_ends, b_ends.detach().cpu().numpy(), axis=0)
                line_starts = np.append(line_starts, l_starts.detach().cpu().numpy(), axis=0)
            # logger.info("example_id_length: %d" % len(example_ids))
        
        if (mode == 'test'):
            attention_file.close()
            global_attention_file.close()
        # eval_accuracy = accuracy(preds,out_label_ids)
        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds_label = np.argmax(preds, axis=1)
        elif args.output_mode == 'questionanswering':
            # Change this to top 5!!!!
            # pred_start_pos = np.argmax(pred_start, axis=1)
            # pred_end_pos = np.argmax(pred_end, axis=1)

            top_indices = np.array(list(map(get_top_k_indices, pred_start, pred_end, [args.topk for i in range(len(pred_start))])))
            # top_indices = get_top_5_indices(pred_start, pred_end)
            # logger.info("top_indices: %s\n" % top_indices)
            pred_start_pos = np.array([[x[0] for x in ex] for ex in top_indices])
            pred_end_pos = np.array([[x[1] for x in ex] for ex in top_indices])
            pred_prob = np.array([[x[2] for x in ex] for ex in top_indices])
            # logger.info("pred_start_pos: %s\n" % pred_start_pos)
            # logger.info("pred_end_pos: %s\n" % pred_end_pos)
            # logger.info("pred_prob: %s\n" % pred_prob)


        # TODO: Group the predictions
        logger.info("length of example_ids: %d", len(example_ids))
        logger.info("example_ids: %s" % (example_ids))
        all_example_ids = np.unique(example_ids)
        logger.info("No of unique example ids: %d" % len(all_example_ids))
        logger.info("unique exampel ids: %s" % (all_example_ids))
        l = [pred_start_pos[example_ids==i] for i in all_example_ids]

        # for ids in token_ids:
        #     logger.info(str(tokenizer.convert_ids_to_tokens(ids)))

        pred_start_by_ex = np.array([pred_start_pos[example_ids==i] for i in all_example_ids])
        pred_end_by_ex = np.array([pred_end_pos[example_ids==i] for i in all_example_ids])
        pred_prob_by_ex = np.array([pred_prob[example_ids==i] for i in all_example_ids])

        token_ids_by_ex = np.array([token_ids[example_ids==i] for i in all_example_ids])
        token_b_start_by_ex = np.array([token_b_starts[example_ids==i] for i in all_example_ids])
        token_b_end_by_ex = np.array([token_b_ends[example_ids==i] for i in all_example_ids])
        line_starts_by_ex = np.array([line_starts[example_ids==i] for i in all_example_ids])
        # start_pos_by_ex = np.array([out_start_pos[example_ids==i] for i in all_example_ids])
        # end_pos_by_ex = np.array([out_end_pos[example_ids==i] for i in all_example_ids])
        
        spans_by_index, labels = file_to_spans_list(input_file_path, args.return_type, args.test_example_begin, args.test_example_end)
        logger.info("pred_start_by_ex: %s, spans_by_index: %s" % (pred_start_by_ex, spans_by_index))
        logger.info("Length of pred_start_by_ex: %d, Length of spans_by_index: %d" % (len(pred_start_by_ex), len(spans_by_index)))
        assert len(pred_start_by_ex) == len(spans_by_index)

        logger.info("Total Example length: %d", len(pred_start_by_ex))
        # logger.info("Pred_start by ex:\n%s", pred_start_by_ex)
        # logger.info("Label by ex:\n%s", labels_by_ex)

        result, selections = compute_metrics(eval_task, tokenizer, token_ids_by_ex, pred_start_by_ex, pred_end_by_ex, pred_prob_by_ex, token_b_start_by_ex, token_b_end_by_ex, line_starts_by_ex, spans_by_index, labels, args.model_type)
        results.update(result)
        if (mode == 'dev'):
            output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
            with open(output_eval_file, "a+") as writer:
                logger.info("***** Eval results {} *****".format(prefix))
                writer.write('evaluate %s\n' % checkpoint)
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))
        elif (mode == 'test'):
            output_test_file = args.test_result_dir
            output_dir = os.path.dirname(output_test_file)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            with open(output_test_file, "w") as writer:
                logger.info("***** Output test results *****")
                # all_logits = preds.tolist()
                # for i, logit in tqdm(enumerate(all_logits), desc='Testing'):
                #     instance_rep = '<CODESPLIT>'.join(
                #         [item.encode('ascii', 'ignore').decode('ascii') for item in instances[i]])

                #     writer.write(instance_rep + '<CODESPLIT>' + '<CODESPLIT>'.join([str(l) for l in logit]) + '\n')
                for key in sorted(result.keys()):
                    print("%s = %s" % (key, str(result[key])))
                    writer.write("%s = %s\n" % (key, str(result[key])))
                for x in selections:
                    # print(x)
                    writer.write("%s\n" % str(x))
            # with open(os.path.join(output_dir, "attention.pkl"), "wb") as attention_file:
            #     pickle.dump(all_attentions, attention_file)
            # with open(os.path.join(output_dir, "global_attention.pkl"), "wb") as global_attention_file:
            #     pickle.dump(all_global_attentions, global_attention_file)
            # logger.info("Saving attention and global attentions to %s", output_dir)

    return results, all_attentions, all_global_attentions

def parse_args():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    return args

def load_and_cache_examples(args, task, tokenizer, ttype='train'):
    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    if ttype == 'train':
        file_name = args.train_file.split('.')[0]
    elif ttype == 'dev':
        file_name = args.dev_file.split('.')[0]
    elif ttype == 'test':
        file_name = args.test_file.split('.')[0]
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}_{}'.format(
        ttype,
        file_name,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))

    # if os.path.exists(cached_features_file):
    try:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        if ttype == 'test':
            final_features = []
            curr_num_examples = 0
            feature_start_index = 0
            while curr_num_examples != args.test_example_begin:
                curr_num_examples = features[feature_start_index + 1].example_id
                feature_start_index += 1
            
            feature_end_index = feature_start_index
            while curr_num_examples != args.test_example_end:
                curr_num_examples = features[feature_end_index + 1].example_id
                feature_end_index += 1
            
            # while curr_num_examples == features[feature_end_index].example_id:
            #     feature_end_index += 1
            # feature_end_index -= 1
            features = features[feature_start_index:feature_end_index]

            logger.info("Feature start and end: %d, %d" % (feature_start_index, feature_end_index))

            examples, instances = processor.get_test_examples(args.data_dir, args.test_file, args.test_example_begin, args.test_example_end)
    except:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if ttype == 'train':
            examples = processor.get_train_examples(args.data_dir, args.train_file)
        elif ttype == 'dev':
            examples = processor.get_dev_examples(args.data_dir, args.dev_file)
        elif ttype == 'test':
            examples, instances = processor.get_test_examples(args.data_dir, args.test_file, args.test_example_begin, args.test_example_end)

        features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode,
                                                cls_token_at_end=bool(args.model_type in ['xlnet']),
                                                # xlnet has a cls token at the end
                                                cls_token=tokenizer.cls_token,
                                                sep_token=tokenizer.sep_token,
                                                cls_token_segment_id=2 if args.model_type in ['xlnet'] else 1,
                                                pad_on_left=bool(args.model_type in ['xlnet']),
                                                # pad on the left for xlnet
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                model_type=args.model_type)

        logger.info("Feature: %s", features[0])
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
    logger.info("No of features: %d" % len(features))
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_example_ids = torch.tensor([f.example_id for f in features], dtype=torch.long)
    all_start_indices = torch.tensor([f.start_index for f in features], dtype=torch.long)
    all_end_indices = torch.tensor([f.end_index for f in features], dtype=torch.long)
    all_line_starts = torch.tensor([f.start_line for f in features], dtype=torch.long)
    all_b_starts = torch.tensor([f.b_start_index for f in features], dtype=torch.long)
    all_b_ends = torch.tensor([f.b_end_index for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    logger.info("Length of all_example_ids: %d" % len(all_example_ids))
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_indices, all_end_indices, all_line_starts, all_b_starts, all_b_ends, all_example_ids)
    if (ttype == 'test'):
        return dataset, instances
    else:
        return dataset

def main(args):
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    # else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    #     torch.cuda.set_device(args.local_rank)
    #     device = torch.device("cuda", args.local_rank)
    #     torch.distributed.init_process_group(backend='nccl')
    #     args.n_gpu = 1
    args.device = device

    set_seed(args)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    
    # label_list = processor.get_labels()
    num_labels = 2

    # Load pretrained model and tokenizer
    args.start_epoch = 0
    args.start_step = 0

    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.model_name_or_path = os.path.join(checkpoint_last, 'pytorch_model.bin')
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels, finetuning_task=args.task_name)
    if args.tokenizer_name:
        tokenizer_name = args.tokenizer_name
    elif args.model_name_or_path:
        tokenizer_name = 'roberta-base'
    tokenizer = tokenizer_class.from_pretrained(tokenizer_name, do_lower_case=args.do_lower_case)
    if args.tokenizer_name and args.tokenizer_name != 'allenai/longformer-base-4096':
        tokenizer.add_tokens(['\n'])
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config)
    model.resize_token_embeddings(len(tokenizer))
    
    # logger.info("Config %s" % config)

    # Distributed and parallel training
    model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # if args.n_gpu > 1:
    #     model = torch.nn.DataParallel(model)
    
    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, ttype='train')
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, optimizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            print(checkpoint)
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, checkpoint=checkpoint, prefix=global_step)[0]
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)

    if args.do_predict:
        print('testing')
        model = model_class.from_pretrained(args.pred_model_dir)
        model.to(args.device)
        evaluate(args, model, tokenizer, checkpoint=None, prefix='', mode='test')
    return results
    
if __name__ == '__main__':
    args = parse_args()
    main(args)