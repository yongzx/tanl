# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import List, Dict
import torch
import logging
import numpy as np
from transformers import PreTrainedTokenizer

from arguments import DataTrainingArguments
from tanl_datasets import load_dataset


def get_avg_results(results: List[dict]) -> dict:
    """
    Compute average results and standard deviation from many episodes.
    """
    aggregate_results = {'num_episodes': len(results)}

    for key in results[0]:
        try:
            numbers = np.array([res[key] for res in results])
            aggregate_results[key] = (numbers.mean(), numbers.std())

        except:
            pass

    return aggregate_results


def print_results(results: dict):
    for key, value in results.items():
        s = f'{key.replace("_", " "):26} '

        if isinstance(value, (list, tuple)):
            mean, std = value
            s += f'{mean:.6f} ± {std:.6f}'
        elif isinstance(value, float):
            s += f'{value:.6f}'
        else:
            s += f'{value}'

        logging.info(s)


def evaluate(model, dataset_name: str, data_args: DataTrainingArguments, tokenizer: PreTrainedTokenizer, split: str,
             seed: int, gpu: int, batch_size: int) -> Dict[str, float]:
    """
    Evaluate a model on some dataset.
    """
    model.eval()

    device = torch.device("cuda", gpu)
    model.to(device)

    logging.info(f'Batch size: {batch_size}')
    logging.info(f'Num beams:  {data_args.num_beams}')
    logging.info(f'Max input length for evaluation:  {data_args.max_seq_length_eval}')
    logging.info(f'Max output length for evaluation: {data_args.max_output_seq_length_eval}')

    test_dataset = load_dataset(
        dataset_name, data_args,
        max_input_length=data_args.max_seq_length_eval,
        max_output_length=data_args.max_output_seq_length_eval,
        tokenizer=tokenizer, split=split, seed=seed, shuffle=False, is_eval=True,
    )

    return test_dataset.evaluate_dataset(data_args=data_args, model=model, device=device, batch_size=batch_size)

def evaluate_no_teacher_forcing_loss(model, dataset_name: str, data_args: DataTrainingArguments, tokenizer: PreTrainedTokenizer, split: str,
             seed: int, gpu: int, batch_size: int) -> Dict[str, float]:
    """
    Evaluate a model on some dataset.
    """
    model.eval()

    device = torch.device("cuda", gpu)
    model.to(device)

    logging.info(f'Batch size: {batch_size}')
    logging.info(f'Num beams:  {data_args.num_beams}')
    logging.info(f'Max input length for evaluation:  {data_args.max_seq_length_eval}')
    logging.info(f'Max output length for evaluation: {data_args.max_output_seq_length_eval}')

    test_dataset = load_dataset(
        dataset_name, data_args,
        max_input_length=data_args.max_seq_length_eval,
        max_output_length=data_args.max_output_seq_length_eval,
        tokenizer=tokenizer, split=split, seed=seed, shuffle=False, is_eval=True,
    )

    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from transformers import default_data_collator
    test_data_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=default_data_collator,
        )

    no_tf_eval_loss = 0.
    count = 0
    for i, inputs in tqdm(enumerate(test_data_loader), total=len(test_data_loader)):
        predictions = model.generate(
            inputs['input_ids'].to(device),
            max_length=data_args.max_output_seq_length_eval,
            num_beams=data_args.num_beams,
            return_dict_in_generate=True,
            output_scores=True
        )
        predictions_logits = torch.transpose(torch.stack(predictions.scores), 0, 1)
        # print(len(predictions.scores), predictions.scores[0].shape)
        # print(len(predictions.sequences), predictions.sequences[0].shape)
        # print(torch.argmax(torch.transpose(torch.stack(predictions.scores), 0, 1), dim=-1))
        # # print(torch.argmax(predictions.scores[0], dim=1))
        # print("="*10)
        # print(predictions.sequences)

        loss_fn = torch.nn.CrossEntropyLoss()
        for (logits, labels) in zip(predictions_logits, inputs['labels']):
            labels = labels[:logits.shape[0]]
            loss = loss_fn(logits, labels.to(device))
            no_tf_eval_loss += loss.item()
            count += 1
    
    return no_tf_eval_loss / count
        
        
