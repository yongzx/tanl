# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import os
import logging
import random
from typing import Dict, Generator, Tuple, List
from abc import ABC, abstractmethod
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer, torch_distributed_zero_first, default_data_collator

from arguments import DataTrainingArguments
from input_example import InputFeatures, InputExample
from input_formats import INPUT_FORMATS
from output_formats import OUTPUT_FORMATS
import collections


class BaseDataset(Dataset, ABC):
    """
    Base class for all datasets.
    """
    name = None         # name of the dataset
    data_name = None    # name of the directory, if different from the name of the dataset
    task_descriptor = None  # string to prepend to every input sentence if multitask=True (default is self.name)

    default_input_format = 'plain'
    default_output_format = None
    default_data_dir = '/users/zyong2/data/zyong2/ws/data/external/tanl_datasets'

    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            max_input_length: int,
            max_output_length: int,
            overwrite_cache: bool = False,
            mode: str = 'train',
            local_rank: int = -1,
            train_subset: float = 1,  # a number < 1 is to use only a subset of training data (random)
            seed: int = None,
            shuffle: bool = True,
            data_args: DataTrainingArguments = None,
            is_eval: bool = False,
    ):
        if seed is not None:
            # set random seed for repeatability
            random.seed(seed)

        self.seed = seed
        self.data_args = data_args
        self.tokenizer = tokenizer

        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        self.input_format = INPUT_FORMATS[
            data_args.input_format if data_args.input_format is not None else self.default_input_format
        ]()
        self.output_format = OUTPUT_FORMATS[
            data_args.output_format if data_args.output_format is not None else self.default_output_format
        ]()

        self.data_path = data_args.data_dir if data_args.data_dir is not None else self.default_data_dir

        self.is_eval = is_eval
        self.eval_nll = data_args.eval_nll

        self.mode = mode
        self.local_rank = local_rank
        self.overwrite_cache = overwrite_cache
        self.train_subset = train_subset
        self.shuffle = shuffle

        self.load_data_init()

    def load_data_init(self):
        cached_data_file = os.path.join(
            self.data_dir(),
            f"cached_{self.name}_{self.mode}_{self.tokenizer.__class__.__name__}_{self.max_input_length}_{self.max_output_length}"
            f"{'_multitask' if self.data_args.multitask else ''}.pth"
        )

        with torch_distributed_zero_first(self.local_rank):
            # make sure only the first process in distributed training processes the dataset,
            # and the others can use the cached version

            if os.path.exists(cached_data_file) and not self.overwrite_cache:
                self.load_cached_data(cached_data_file)

            else:
                self.load_schema()   # here the dataset can load information such as entity/relation types
                self.examples = self.load_data(mode=self.mode, seed=self.seed)

                # assign examples to this dataset
                for example in self.examples:
                    example.dataset = self

                self.features = self.compute_features(
                    max_input_length=self.max_input_length,
                    max_output_length=self.max_output_length,
                    multitask=self.data_args.multitask,
                )

                if self.local_rank in [-1, 0]:
                    # save data
                    self.save_data(cached_data_file)

            # shuffle indices
            self.indices = list(range(len(self.examples)))
            if self.seed is not None and self.shuffle:
                random.shuffle(self.indices)

            # compute effective size of the dataset
            self.effective_size = round(self.train_subset * len(self.examples))
            if self.train_subset != 1:
                logging.info(f"Effective dataset size reduced to {self.effective_size} ({self.train_subset * 100:.0f}%)")

    def __repr__(self):
        return f'Dataset {self.name}'

    def __len__(self):
        return self.effective_size

    def __getitem__(self, i: int) -> InputFeatures:
        return self.features[self.indices[i]]

    def get_example(self, i: int) -> InputExample:
        return self.examples[self.indices[i]]

    def data_dir(self):
        if self.data_name is not None:
            return os.path.join(self.data_path, self.data_name)
        else:
            return os.path.join(self.data_path, self.name)

    def load_cached_data(self, cached_data_file: str):
        d = torch.load(cached_data_file)
        self.examples, self.features = d['examples'], d['features']

    def save_data(self, cached_data_file: str):
        torch.save({
            'examples': self.examples,
            'features': self.features,
        }, cached_data_file)

    def load_schema(self):
        """
        Load extra dataset information, such as entity/relation types.
        """
        pass

    @abstractmethod
    def load_data_single_split(self, split: str, seed: int = None) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).
        """
        pass

    def load_data(self, mode: str, seed: int = None) -> List[InputExample]:
        """
        Load all data, where 'mode' is a list of comma-separated splits to use.
        """
        examples = []

        if isinstance(mode, str):
            splits = mode.split(',')
        else:
            assert isinstance(mode, (list, tuple))
            splits = mode

        for split in splits:
            examples += self.load_data_single_split(split, seed=seed)

        return examples

    def _warn_max_sequence_length(self, max_sequence_length: int, sentences: List[str], name: str):
        max_length_needed = max(len(self.tokenizer.tokenize(x)) for x in sentences)
        if max_length_needed > max_sequence_length:
            logging.warning(
                f'Max sequence length is {max_sequence_length} but the longest {name} sequence is '
                f'{max_length_needed} long'
            )

    def compute_features(self, max_input_length: int, max_output_length: int, multitask: bool = False):
        input_sentences = [self.input_format.format_input(example, multitask=multitask) for example in self.examples]
        output_sentences = [self.output_format.format_output(example)['output_sentence'] for example in self.examples]

        input_tok = self.tokenizer.batch_encode_plus(
            input_sentences,
            max_length=max_input_length,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
        )
        self._warn_max_sequence_length(max_input_length, input_sentences, "input")

        output_tok = self.tokenizer.batch_encode_plus(
            output_sentences,
            max_length=max_output_length,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
        )
        self._warn_max_sequence_length(max_output_length, output_sentences, "output")

        assert input_tok.input_ids.size(0) == output_tok.input_ids.size(0)
    
        features = []
        for sentence_input_ids, att_mask, label_input_ids in zip(input_tok.input_ids, input_tok.attention_mask,
                                                                 output_tok.input_ids):
            features.append(InputFeatures(
                input_ids=sentence_input_ids.tolist(),
                attention_mask=att_mask.tolist(),
                label_ids=label_input_ids.tolist()
            ))
    
        return features

    def generate_output_sentences(self, data_args: DataTrainingArguments, model, device, batch_size: int) \
            -> Generator[Tuple[InputExample, str], None, None]:
        """
        Generate pairs (example, output_sentence) for evaluation.
        """
        test_data_loader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=default_data_collator,
        )

        for i, inputs in tqdm(enumerate(test_data_loader), total=len(test_data_loader)):
            predictions = model.generate(
                inputs['input_ids'].to(device),
                max_length=data_args.max_output_seq_length_eval,
                num_beams=data_args.num_beams,
            )

            for j, (input_ids, label_ids, prediction) in enumerate(
                    zip(inputs['input_ids'], inputs['labels'], predictions)):
                current_id = i * batch_size + j
                example = self.get_example(current_id)
                output_sentence = self.tokenizer.decode(prediction, skip_special_tokens=True,
                                                        clean_up_tokenization_spaces=False)

                yield example, output_sentence

    @abstractmethod
    def evaluate_dataset(self, data_args: DataTrainingArguments, model, device, batch_size: int, macro: bool = False) \
            -> Dict[str, float]:
        """
        Evaluate model on this dataset, returning the task-relevant metrics.
        """
        pass

######################################################################################################################################################################################
######################################################################################################################################################################################
######################################################################################################################################################################################
######################################################################################################################################################################################
######################################################################################################################################################################################
######################################################################################################################################################################################
######################################################################################################################################################################################
######################################################################################################################################################################################
######################################################################################################################################################################################
######################################################################################################################################################################################


class NoisyBaseDataset(BaseDataset):
    def __init__(self, noisy_dir_name, inputs_fp, viterbi_paths_fp, viterbi_scores_fp, top_k_noisy_seq,
                 *args, **kwargs):
        self.noisy_dir_name = noisy_dir_name
        self.inputs_fp = inputs_fp
        self.viterbi_paths_fp = viterbi_paths_fp
        self.viterbi_scores_fp = viterbi_scores_fp
        self.top_k_noisy_seq = top_k_noisy_seq
        super().__init__(*args, **kwargs)
        
    def load_data_init(self):
        if self.mode == "train":
            cached_data_file = os.path.join(
                self.data_dir(),
                f"cached_{self.name}_noise{self.noisy_dir_name}_{self.mode}_{self.tokenizer.__class__.__name__}_{self.max_input_length}_{self.max_output_length}"
                f"{'_multitask' if self.data_args.multitask else ''}.pth"
            )

            with torch_distributed_zero_first(self.local_rank):
                # make sure only the first process in distributed training processes the dataset,
                # and the others can use the cached version

                if os.path.exists(cached_data_file) and not self.overwrite_cache:
                    self.load_cached_data(cached_data_file)

                else:
                    self.load_schema()   # here the dataset can load information such as entity/relation types
                    self.examples = self.load_data(mode=self.mode, seed=self.seed, 
                                                   inputs_fp=self.inputs_fp, viterbi_paths_fp=self.viterbi_paths_fp,
                                                   viterbi_scores_fp=self.viterbi_scores_fp)

                    # assign examples to this dataset
                    for example in self.examples:
                        example.dataset = self

                    self.features = self.compute_features(
                        max_input_length=self.max_input_length,
                        max_output_length=self.max_output_length,
                        multitask=self.data_args.multitask,
                    )

                    if self.local_rank in [-1, 0]:
                        # save data
                        self.save_data(cached_data_file)

                # deduplicate self.examples
                dedup_examples = []
                for i, example in enumerate(self.examples):
                    if self.clean_bool[i]:
                        dedup_examples.append(example)
                self.examples = dedup_examples
                logging.info(f"🧽 Deduplicated to {len(self.examples)} examples for split {self.mode}")

                # shuffle indices
                self.indices = list(range(len(self.examples)))
                if self.seed is not None and self.shuffle:
                    random.shuffle(self.indices)
                    logging.info(f"🃏 Shuffling the examples for split {self.mode}")
                else:
                    logging.info(f"❌ No shuffling for split {self.mode}")

                # compute effective size of the dataset
                self.effective_size = round(self.train_subset * len(self.examples))
                if self.train_subset != 1:
                    logging.info(f"Effective dataset size reduced to {self.effective_size} ({self.train_subset * 100:.0f}%)")
        else:
            super().load_data_init()

    def load_data(self, mode: str, seed: int = None, inputs_fp: str = None, viterbi_paths_fp: str = None, viterbi_scores_fp: str = None) -> List[InputExample]:
        """
        Load all data, where 'mode' is a list of comma-separated splits to use.
        """
        examples = []

        if isinstance(mode, str):
            splits = mode.split(',')
        else:
            assert isinstance(mode, (list, tuple))
            splits = mode

        for split in splits:
            examples += self.load_data_single_split(split, seed=seed, 
                                                    inputs_fp=inputs_fp, viterbi_paths_fp=viterbi_paths_fp, viterbi_scores_fp=viterbi_scores_fp)

        return examples
    
    def compute_features(self, max_input_length: int, max_output_length: int, multitask: bool = False):
        # called after self.load_data
        # output is stored at self.features

        input_sentences = [self.input_format.format_input(example, multitask=multitask) for example in self.examples]
        output_sentences = [self.output_format.format_output(example)['output_sentence'] for example in self.examples]

        input_tok = self.tokenizer.batch_encode_plus(input_sentences,max_length=max_input_length,return_tensors='pt',padding='max_length',truncation=True)
        self._warn_max_sequence_length(max_input_length, input_sentences, "input")

        output_tok = self.tokenizer.batch_encode_plus(output_sentences,max_length=max_output_length,return_tensors='pt',padding='max_length',truncation=True)
        self._warn_max_sequence_length(max_output_length, output_sentences, "output")
        
        # map input tokens to noisy output ids and their weights and total LSE
        if self.mode == "train":
            gold_output_sentences = [self.output_format.format_output(example)['gold_output_sentence'] for example in self.examples]
            self.clean_bool = list() # purpose: remove duplicates
            self.INPUT_IDS_TO_OUTPUT_IDS = collections.defaultdict(list)
            self.INPUT_IDS_TO_WEIGHTS = collections.defaultdict(list)
            self.INPUT_IDS_TO_GOLD = dict()
            for i in range(len(self.examples)):
                input_ids = tuple(input_tok['input_ids'][i].tolist())
                if input_ids not in self.INPUT_IDS_TO_OUTPUT_IDS:
                    self.clean_bool.append(True)
                else:
                    self.clean_bool.append(False)

                self.INPUT_IDS_TO_OUTPUT_IDS[input_ids].append(output_tok['input_ids'][i])
                self.INPUT_IDS_TO_WEIGHTS[input_ids].append((self.examples[i].noise_weight, self.examples[i].total_LSE_noise_weight))
                self.INPUT_IDS_TO_GOLD[input_ids] = gold_output_sentences[i]

            ### NOTE: ontonotes, ncbi has duplicate inputs
            # for v in self.INPUT_IDS_TO_OUTPUT_IDS.values():
            #     print(len(v))

            features = []
            for i, (sentence_input_ids, att_mask, label_input_ids) in enumerate(zip(input_tok.input_ids, input_tok.attention_mask,
                                                                 output_tok.input_ids)):
                if self.clean_bool[i]:
                    label_ids = label_input_ids.tolist()
                    features.append(InputFeatures(
                        input_ids=sentence_input_ids.tolist(),
                        attention_mask=att_mask.tolist(),
                        label_ids=label_ids
                    ))

            return features

        assert input_tok.input_ids.size(0) == output_tok.input_ids.size(0)

        #### original TANL: get features
        features = []
        for sentence_input_ids, att_mask, label_input_ids in zip(input_tok.input_ids, input_tok.attention_mask,
                                                                 output_tok.input_ids):
            label_ids = label_input_ids.tolist()
            features.append(InputFeatures(
                input_ids=sentence_input_ids.tolist(),
                attention_mask=att_mask.tolist(),
                label_ids=label_ids
            ))

        # ### commented out (useful when we treat each noisy output as a single instance)
        # ### if noisy

        # self.OUTPUT_IDS_TO_EXAMPLES = dict()
        # for i in range(len(self.examples)):
        #     self.OUTPUT_IDS_TO_EXAMPLES[tuple(output_tok['input_ids'][i].tolist())] = self.examples[i]

        # example_noise_weights = list()
        # example_total_LSE_noise_weights = list()
        # for example in self.examples:
        #     if example.noise_weight is not None and example.total_LSE_noise_weight is not None:
        #         example_noise_weights.append(example.noise_weight)
        #         example_total_LSE_noise_weights.append(example.total_LSE_noise_weight)

        # if not self.is_eval and self.examples[0].gold_entities is not None: # heuristic if-condition
        #     gold_output_sentences = [self.output_format.format_output(example)['gold_output_sentence'] for example in self.examples]
        
        # if example_total_LSE_noise_weights:
        #     self.NOISY_OUTPUT_IDS_TO_WEIGHTS = dict()
        #     self.NOISY_OUTPUT_IDS_TO_GOLD = dict()
        #     for i, input_ids in enumerate(output_tok['input_ids']):
        #         self.NOISY_OUTPUT_IDS_TO_WEIGHTS[tuple(input_ids.tolist())] = (example_noise_weights[i], example_total_LSE_noise_weights[i])
        #         self.NOISY_OUTPUT_IDS_TO_GOLD[tuple(input_ids.tolist())] = gold_output_sentences[i]
        # else:
        #     self.NOISY_OUTPUT_IDS_TO_WEIGHTS = None
        return features
        