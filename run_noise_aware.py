# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Copyright 2020 The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Uses some code from
# https://github.com/huggingface/transformers/blob/master/examples/seq2seq/finetune_trainer.py


import argparse
import configparser
import itertools
import json
import logging
import os
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, AutoModelForSeq2SeqLM, Trainer

from arguments import ModelArguments, DataTrainingArguments, TrainingArguments, NoiseAwareArguments
from tanl_datasets import load_dataset
from evaluate import evaluate, get_avg_results, print_results, evaluate_no_teacher_forcing_loss
from utils import get_episode_indices, get_precision_recall_f1
from transformers.file_utils import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    get_full_repo_name,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
)


# from tensorboardX import SummaryWriter
import wandb

def main():
    assert torch.cuda.is_available(), 'CUDA not available'

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('job')
    parser.add_argument('-c', '--config_file', type=str, default='/users/zyong2/data/zyong2/ws/data/external/tanl/config.ini', help='configuration file')
    parser.add_argument('-e', '--eval', action='store_true', default=False, help='run evaluation only')
    parser.add_argument('--evaluate_checkpoints', action='store_true', default=False,
                        help='evaluate intermediate checkpoints instead of the final model')
    parser.add_argument('--evaluate_last_checkpoint', action='store_true', default=False,
                        help='evaluate the last intermediate checkpoint instead of the final model')
    parser.add_argument('--evaluate_checkpoint_in_dir', type=str, default=None,
                        help='evaluate the checkpoint in the given directory')
    parser.add_argument('-a', '--evaluate_all', action='store_true', default=False,
                        help='evaluate intermediate checkpoints together with the final model')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='which GPU to use for evaluation')
    parser.add_argument('-v', '--verbose_results', action='store_true', default=False,
                        help='print results for each evaluation run')
    parser.add_argument('--debug_loss_flag', action='store_true', default=False,
                        help='set the debug_flag in noise aware loss to True')
    parser.add_argument('--use_accum', action='store_true', default=False,
                        help='use custom gradient accumulation in the loss function')
    args, remaining_args = parser.parse_known_args()
    
    # read config file
    config = configparser.ConfigParser(allow_no_value=False)
    config.read(args.config_file)
    job = args.job

    assert job in config

    # set defaults for other arguments
    defaults = {
        'overwrite_output_dir': True,
        'overwrite_cache': True,
        'per_device_eval_batch_size': 4,
        'learning_rate': 5e-4,
        'logging_steps': 2000,     # do not log by default
        'save_steps': 2000,        # do not save checkpoints by default
        'eval_steps': 2000,
        'evaluation_strategy': "steps",
        "save_strategy": "steps",
    }

    # the config file gives default values for the command line arguments
    defaults.update(dict(config.items(job)))
    for key in defaults:
        if defaults[key] in ['True', 'False']:
            # interpret True/False as boolean
            defaults[key] = config.getboolean(job, key)
        if defaults[key] == 'None':
            # interpret as None
            defaults[key] = None

    if args.eval:
        # run evaluation only
        defaults['do_train'] = False

    # parse remaining arguments and divide them into three categories
    second_parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, NoiseAwareArguments))
    second_parser.set_defaults(**defaults)
    print(defaults)
    print(second_parser.dataclass_types[2])
    model_args, data_args, training_args, noise_aware_args = second_parser.parse_args_into_dataclasses(remaining_args)
    print("=== model_args ===")
    print(model_args)
    print("=== data_args ===")
    print(data_args)
    print("=== training_args ===")
    print(training_args)
    print("=== noise_aware_args ===")
    print(noise_aware_args)

    try:
        os.mkdir(training_args.output_dir)
    except FileExistsError:
        pass

    # process arguments related to max length
    if data_args.max_output_seq_length_eval is None:
        # defaults first to max_output_seq_length, then max_seq_length_eval, then max_seq_length
        data_args.max_output_seq_length_eval = data_args.max_output_seq_length \
                                               or data_args.max_seq_length_eval \
                                               or data_args.max_seq_length

    if data_args.max_output_seq_length is None:
        # defaults to max_seq_length
        data_args.max_output_seq_length = data_args.max_seq_length

    if data_args.max_seq_length_eval is None:
        # defaults to max_seq_length
        data_args.max_seq_length_eval = data_args.max_seq_length

    if data_args.chunk_size_eval is None:
        # defaults to chunk_size
        data_args.chunk_size_eval = data_args.chunk_size

    if data_args.chunk_overlap_eval is None:
        # defaults to chunk overlap
        data_args.chunk_overlap_eval = data_args.chunk_overlap

    # construct name for the output directory
    # for example: conll04-t5-base-ep200-len256-ratio0-b4-train
    output_dir = os.path.join(
        training_args.output_dir,
        f'{args.job}'
        f'-noise_seq_{noise_aware_args.noisy_dir_name}'
        f'-{model_args.model_name_or_path.split("/")[-1]}'
        f'-ep{round(training_args.num_train_epochs)}'
        f'-len{data_args.max_seq_length}'
    )

    if data_args.max_output_seq_length != data_args.max_seq_length:
        output_dir += f'-{data_args.max_output_seq_length}'

    if data_args.data_start is not None:
        output_dir += f'-start{data_args.data_start}'

    if data_args.data_end is not None:
        output_dir += f'-end{data_args.data_end}'

    if training_args.learning_rate != 5e-4:
        output_dir += f'-lr{training_args.learning_rate}'

    output_dir += f'-b{training_args.per_device_train_batch_size}' \
                  f'-{data_args.train_split}'

    if data_args.chunk_size != 128:
        output_dir += f'-chunk{data_args.chunk_size}'
    if data_args.chunk_overlap != 64:
        output_dir += f'-overlap{data_args.chunk_overlap}'

    if data_args.output_format is not None:
        output_dir += f'-{data_args.output_format}'
    if data_args.input_format is not None:
        output_dir += f'-{data_args.input_format}'
    if data_args.train_subset < 1:
        output_dir += f'-size{data_args.train_subset:.2f}'

    try:
        os.mkdir(output_dir)
    except FileExistsError:
        pass

    # setup logging
    logging.basicConfig(
      filename=os.path.join(output_dir, 'logs.log'),
      format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
      datefmt='%Y-%m-%d %H:%M:%S',
      level=logging.INFO,
    )
    logging.getLogger().addHandler(logging.StreamHandler())

    # construct file name for the evaluation results
    evaluation_output_filename = f'results'
    if data_args.num_beams is not None:
        evaluation_output_filename += f'-{data_args.num_beams}beams'
    if data_args.max_seq_length_eval is not None:
        evaluation_output_filename += f'-len{data_args.max_seq_length_eval}'

    # create model config
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    # create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    )

    # get list of dataset names
    dataset_names = data_args.datasets.split(',')

    # construct list of episode indices
    episode_indices = get_episode_indices(data_args.episodes)

    # episode loop
    # (note that the episode index is used as the random seed, so that each episode is reproducible)
    evaluation_results = defaultdict(list)
    for ep_idx in episode_indices:
        print()
        logging.info(f'Episode {ep_idx} ({len(episode_indices)} episodes total)')
        episode_output_dir = os.path.join(output_dir, f'episode{ep_idx}')

        try:
            os.mkdir(episode_output_dir)
        except FileExistsError:
            pass

        logging.info(f'Output directory: {episode_output_dir}')

        training_args.output_dir = episode_output_dir   # checkpoints are saved in episode-specific directory

        # load pretrained model
        model = None
        if training_args.resume_from_checkpoint:
            logging.info(f"Resume from checkpoint {training_args.resume_from_checkpoint}")
            model = AutoModelForSeq2SeqLM.from_pretrained(
                training_args.resume_from_checkpoint,
                config=config,
                cache_dir=model_args.cache_dir,
            )
        elif training_args.zero_shot or training_args.do_train:
            logging.info(f"Using model {model_args.model_name_or_path}")
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=model_args.cache_dir,
            )

        # fine-tune the model
        if training_args.do_train:
            # load train dataset
            datasets = []
            for dataset_name in dataset_names:
                logging.info(f'Process noisy dataset {dataset_name} (train)')
                noise_dataset = load_dataset(
                    dataset_name, data_args, noise_aware_args=noise_aware_args, 
                    split=data_args.train_split,
                    max_input_length=data_args.max_seq_length, max_output_length=data_args.max_output_seq_length,
                    tokenizer=tokenizer, seed=ep_idx, train_subset=data_args.train_subset, 
                    noisy_dir_name=noise_aware_args.noisy_dir_name,
                    inputs_fp=noise_aware_args.inputs_fp,
                    viterbi_paths_fp=noise_aware_args.viterbi_paths_fp,
                    viterbi_scores_fp=noise_aware_args.viterbi_scores_fp,
                    top_k_noisy_seq=noise_aware_args.top_k_noisy_seq,
                    load_noisy=True,
                    data_start=0,
                    data_end=1
                )
                datasets.append(noise_dataset) # FIXME: noise_dataset is called later on in loss function
            train_dataset = torch.utils.data.ConcatDataset(datasets) if training_args.do_train else None

            datasets = []
            for dataset_name in dataset_names:
                logging.info(f'Process dataset {dataset_name} (dev)')
                dev_dataset = load_dataset(
                    dataset_name, data_args, noise_aware_args=noise_aware_args, split=data_args.val_split,
                    max_input_length=data_args.max_seq_length_eval,
                    max_output_length=data_args.max_output_seq_length_eval,
                    tokenizer=tokenizer, seed=ep_idx, shuffle=False, is_eval=True
                )
                datasets.append(dev_dataset) # FIXME: dev_dataset is called later on.
            eval_dataset = torch.utils.data.ConcatDataset(datasets) if training_args.do_train else None
                
            class NoiseAwareTrainer(Trainer):
                def __init__(self, *trainer_args, **trainer_kwargs):
                    super().__init__(*trainer_args, **trainer_kwargs)
                    self.debug_loss_flag = args.debug_loss_flag
                    self.use_accum = args.use_accum

                def compute_custom_loss(self, model, inputs, return_outputs=False):
                    """
                    How the loss is computed by Trainer. By default, all models return the loss in the first element.

                    Subclass and override for custom behavior.
                    """
                    device = model.device if not isinstance(model, torch.nn.DataParallel) else model.module.device
                    total_loss = 0.
                    for i in range(inputs['input_ids'].shape[0]):
                        input_ids = tuple(inputs['input_ids'][i].tolist())
                        
                        # get noisy outputs and their corresponding weights
                        output_ids = noise_dataset.INPUT_IDS_TO_OUTPUT_IDS[input_ids]
                        raw_weights = noise_dataset.INPUT_IDS_TO_WEIGHTS[input_ids]
                        weights, total_LSE_weights = zip(*raw_weights)
                        weights, total_LSE_weights = torch.Tensor(weights).to(device), torch.Tensor(total_LSE_weights).to(device)
                        output_ids = torch.stack(output_ids)

                        noise_aware_loss = 0.
                        bsz = noise_aware_args.bsz
                        for j in range(len(output_ids) // bsz + 1):
                            if j * bsz >= len(output_ids):
                                break
                            labels = output_ids[(j * bsz):((j + 1) * bsz)].to(device)
                            outputs = model(input_ids=inputs['input_ids'][i].repeat(labels.size(0), 1).to(device), 
                                            attention_mask=inputs['attention_mask'][i].repeat(labels.size(0), 1).to(device),
                                            labels=labels,
                                            return_dict=True)   
                            lm_logits = outputs.logits.to(device)
                            
                            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
                            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))      
                            loss = torch.mean(torch.stack(torch.split(loss, split_size_or_sections=labels.size(-1))), dim=-1)
                            noise_aware_loss += torch.sum(torch.exp(weights[(j * bsz):((j + 1) * bsz)] + torch.log(loss) - total_LSE_weights[(j * bsz):((j + 1) * bsz)]))
                            
                            if (j > 0 and (j * bsz) % noise_aware_args.when_call_backward == 0):
                                # print("noise_aware_loss:", noise_aware_loss)
                                noise_aware_loss /= (self.args.gradient_accumulation_steps * inputs['input_ids'].shape[0])
                                # noise_aware_loss /= backward_count  # the loss is already scaled by the weighted loss (https://stackoverflow.com/questions/62067400/understanding-accumulated-gradients-in-pytorch)
                                noise_aware_loss.backward()
                                total_loss += noise_aware_loss
                                noise_aware_loss = 0.  # reset noise_aware_loss

                            if self.debug_loss_flag:
                                print(f"{f'Example {i} ({j}-th noisy instance)':-<100}")
                                print(f"training input_ids:", tokenizer.decode(inputs['input_ids'][i].tolist(), skip_special_tokens=True))
                                print(f"training gold ⭐️ sentence:", noise_dataset.INPUT_IDS_TO_GOLD[input_ids])
                                print(f"training noisy 💥 label_ids:", tokenizer.decode(output_ids[j].tolist(), skip_special_tokens=True))
                                
                                if isinstance(model, torch.nn.DataParallel):
                                    print(f"model inference output 👉:", tokenizer.decode(model.module.generate(torch.unsqueeze(inputs['input_ids'][i], dim=0))[0], skip_special_tokens=True))
                                else:
                                    print(f"model inference output 👉:", tokenizer.decode(model.generate(torch.unsqueeze(inputs['input_ids'][i], dim=0))[0], skip_special_tokens=True))
                                
                                print("training input_weight:" )
                                # print(f"weighted noise_aware_loss = {torch.exp(weight + torch.log(outputs['loss']) - total_LSE_weight).item()}\n"
                                #     f"output.loss = {outputs.loss.item()} with weight = {weight} and total LSE weight = {total_LSE_weight}")
                                print(weights[(j * bsz):((j + 1) * bsz)])
                        
                        if noise_aware_loss != 0.:
                            noise_aware_loss /= (self.args.gradient_accumulation_steps * inputs['input_ids'].shape[0])
                            noise_aware_loss.backward()
                            total_loss += noise_aware_loss
                            noise_aware_loss = 0.  # reset noise_aware_loss
                    
                        # if self.debug_loss_flag and j > 5:
                        #     print("Finish printing one batch")
                        #     assert False


                    return total_loss

                def training_step(self, model, inputs):
                    model.train()
                    inputs = self._prepare_inputs(inputs)

                    if is_sagemaker_mp_enabled():
                        scaler = self.scaler if self.do_grad_scaling else None
                        loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps, scaler=scaler)
                        return loss_mb.reduce_mean().detach().to(self.args.device)

                    with self.autocast_smart_context_manager():
                        loss = self.compute_custom_loss(model, inputs)

                    if self.args.n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu parallel training

                    # if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
                    #     # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
                    #     loss = loss / self.args.gradient_accumulation_steps

                    # if self.do_grad_scaling:
                    #     self.scaler.scale(loss).backward()
                    # elif self.use_apex:
                    #     with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    #         scaled_loss.backward()
                    # elif self.deepspeed:
                    #     # loss gets scaled under gradient_accumulation_steps in deepspeed
                    #     loss = self.deepspeed.backward(loss)
                    # else:
                    #     loss.backward()
                    # print("Loss:", loss)
                    # assert False

                    return loss.detach()

            if "wandb" in training_args.report_to:
                wandb.init(name=training_args.run_name, settings=wandb.Settings(start_method='fork'))

            trainer = NoiseAwareTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset
            )

            # start trainer
            logging.info('Start training')
            trainer.train()

            # save model parameters
            trainer.save_model(episode_output_dir)
        
        # run evaluation
        if training_args.local_rank in [-1, 0] and (training_args.do_eval or training_args.do_predict):
            # should we evaluate on dev, test, or both?

            # Transformer.TrainingArguments will set do_eval to `True` if `evaluation_strategy` is different from `"no"`
            # Therefore, we have to manually switch it off in this evaluation setting where we use the do_eval value in the config.ini file.
            if not defaults['do_eval']:
                training_args.do_eval = False

            evaluation_splits = []
            if training_args.do_eval:
                evaluation_splits.append('dev')
            if training_args.do_predict:
                evaluation_splits.append('test')

            if args.evaluate_checkpoints:
                evaluation_dirs = list(sorted([
                        checkpoint_dir
                        for checkpoint_dir in os.listdir(episode_output_dir)
                        if checkpoint_dir.startswith('checkpoint-')
                    ], key=lambda x: int(x[len('checkpoint-'):])))

                if data_args.eval_datasets is None:
                    eval_dataset_names = dataset_names
                else:
                    eval_dataset_names = data_args.eval_datasets.split(',')
                
                best_ckpt_dir = None
                best_f1_score = float('-inf')
                train_results = list()
                all_results = list()
                all_no_tf_loss = list()

                if training_args.do_eval_train:
                    for comb in itertools.product(['train'], evaluation_dirs, eval_dataset_names):
                        split, evaluation_dir, dataset_name = comb
                        model_dir = os.path.join(episode_output_dir, evaluation_dir)
                        model = AutoModelForSeq2SeqLM.from_pretrained(
                                model_dir,
                                config=config,
                            )
            
                        logging.info(f"{f'[Train data] 🏃🏻‍♂️ Evaluating {evaluation_dir}':=^100}")
                        res = evaluate(
                            model=model, dataset_name=dataset_name, data_args=data_args, noise_aware_args=noise_aware_args, tokenizer=tokenizer, split=split,
                            seed=ep_idx, batch_size=training_args.per_device_eval_batch_size, gpu=args.gpu, load_noisy=False
                        )
                        train_results.append(res['entity_f1'])

                    if "wandb" in training_args.report_to:
                        wandb.init(name=training_args.run_name, reinit=True, settings=wandb.Settings(start_method='fork'))
                        for n_iter in range(int(training_args.num_train_epochs)):
                            wandb.log({"eval/train_post_f1": train_results[n_iter], # "eval/no_tf_loss":  all_no_tf_loss[n_iter],
                                    "eval/post_global_steps": int(evaluation_dirs[n_iter].split('-')[-1])})
                    assert False
                    
                # use the dev set to select the best checkpoint
                for comb in itertools.product(['dev'], evaluation_dirs, eval_dataset_names):
                    split, evaluation_dir, dataset_name = comb
                    model_dir = os.path.join(episode_output_dir, evaluation_dir)
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                            model_dir,
                            config=config,
                        )
                    
                    logging.info(f"{f'[Dev data] 🏃🏻‍♂️ Evaluating {evaluation_dir}':=^100}")
                    res = evaluate(
                        model=model, dataset_name=dataset_name, data_args=data_args, noise_aware_args=noise_aware_args, tokenizer=tokenizer, split=split,
                        seed=ep_idx, batch_size=training_args.per_device_eval_batch_size, gpu=args.gpu
                    )

                    # no_tf_loss = evaluate_no_teacher_forcing_loss(
                    #     model=model, dataset_name=dataset_name, data_args=data_args, tokenizer=tokenizer, split=split,
                    #     seed=ep_idx, batch_size=training_args.per_device_eval_batch_size, gpu=args.gpu
                    # )

                    all_results.append(res['entity_f1'])
                    # all_no_tf_loss.append(no_tf_loss)
                    

                    if res['entity_f1'] > best_f1_score:
                        best_f1_score = res['entity_f1']
                        best_ckpt_dir = evaluation_dir
                
                # # write to tensorboard
                # writer = SummaryWriter(logdir=f"{training_args.logging_dir}")
                # for n_iter in range(int(training_args.num_train_epochs)):
                #     writer.add_scalar('eval/f1', all_results[n_iter], n_iter)

                if "wandb" in training_args.report_to:
                    wandb.init(name=training_args.run_name, reinit=True, settings=wandb.Settings(start_method='fork'))
                    for n_iter in range(int(training_args.num_train_epochs)):
                        wandb.log({"eval/dev_post_f1": all_results[n_iter], # "eval/no_tf_loss":  all_no_tf_loss[n_iter],
                                "eval/post_global_steps": int(evaluation_dirs[n_iter].split('-')[-1])})
                    
                if best_ckpt_dir is not None:
                    print(f"🔥 choosing checkpoint {best_ckpt_dir} with dev-score {best_f1_score}")
                    args.evaluate_checkpoint_in_dir = best_ckpt_dir

            # should we evaluate on the final model and/or on all intermediate checkpoints?
            evaluation_dirs = []

            if args.evaluate_checkpoints or args.evaluate_last_checkpoint or \
                    args.evaluate_checkpoint_in_dir or args.evaluate_all:
                # all intermediate checkpoints
                evaluation_dirs = list(sorted([
                    checkpoint_dir
                    for checkpoint_dir in os.listdir(episode_output_dir)
                    if checkpoint_dir.startswith('checkpoint-')
                ], key=lambda x: int(x[len('checkpoint-'):])))
                if args.evaluate_last_checkpoint:
                    # only evaluate on the last checkpoint
                    evaluation_dirs = [evaluation_dirs[-1]]
                elif args.evaluate_checkpoint_in_dir:
                    assert args.evaluate_checkpoint_in_dir in evaluation_dirs, \
                        "checkpoint {} does not exist".format(args.evaluate_checkpoint_in_dir)
                    evaluation_dirs = [args.evaluate_checkpoint_in_dir]
                    print(f"evaluate_checkpoint_in_dir 🔖: {args.evaluate_checkpoint_in_dir}")

            if args.evaluate_all or (not args.evaluate_checkpoints and not args.evaluate_last_checkpoint):
                # evaluate on the final model
                evaluation_dirs += ['']

            # datasets to evaluate on
            if data_args.eval_datasets is None:
                eval_dataset_names = dataset_names
            else:
                eval_dataset_names = data_args.eval_datasets.split(',')

            # evaluate all possible combinations of dev/test, model, and datasets
            print("evaluation_dirs:", evaluation_dirs)
            print("evaluation_splits:", evaluation_splits)
            assert False
            for comb in itertools.product(evaluation_splits, evaluation_dirs, eval_dataset_names):
                split, evaluation_dir, dataset_name = comb
                model_dir = os.path.join(episode_output_dir, evaluation_dir)

                if args.evaluate_checkpoints or args.evaluate_last_checkpoint or args.evaluate_all or model is None:
                    # we need to load the model
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_dir,
                        config=config,
                    )

                if len(evaluation_dir) > 0:
                    logging.info(f"{f'✅🎯 Checkpoint {evaluation_dir} evaluated on {dataset_name} {split}':-<100}")
                else:
                    logging.info(f'Evaluate on {dataset_name} {split}')

                res = evaluate(
                    model=model, dataset_name=dataset_name, data_args=data_args, noise_aware_args=noise_aware_args, tokenizer=tokenizer, split=split,
                    seed=ep_idx, batch_size=training_args.per_device_eval_batch_size, gpu=args.gpu, load_noisy=False
                )
                # store results
                evaluation_results[comb].append(res)

                # print results
                if args.verbose_results:
                    print_results(res)

                # save results to file
                with open(
                        os.path.join(model_dir, evaluation_output_filename + f'-{dataset_name}-{split}.json'), 'w'
                ) as f:
                    json.dump(res, f, indent=0)

    # print average results and save them to file
    for comb, results in evaluation_results.items():
        split, evaluation_dir, dataset_name = comb

        print()
        logging.info(
            f'Average of {split} results over {len(results)} episodes ({dataset_name} {evaluation_dir}):'
        )
        res = get_avg_results(results)

        # print average results
        print_results(res)

        # save average results to file
        filename = evaluation_output_filename + f'-{dataset_name}-{split}'
        if len(evaluation_dir) > 0:
            filename += '-'
        filename += f'{evaluation_dir}.json'

        with open(os.path.join(output_dir, filename), 'w') as f:
            json.dump(res, f, indent=0)

    logging.info(f'Model weights and intermediate checkpoints saved in {output_dir}')


if __name__ == "__main__":
    main()
