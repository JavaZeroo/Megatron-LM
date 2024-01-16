# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""GPT style dataset."""

import os
import time

import numpy as np
import torch

from megatron import print_rank_0, get_args
from megatron.core import mpu
from megatron.data.blendable_dataset import BlendableDataset
from megatron.data.dataset_utils import get_datasets_weights_and_num_samples
from megatron.data.dataset_utils import get_train_valid_test_split_
from dataset_conv import get_processed_dataset, load_incontext_fewshot_samples
from dataset_conv import get_fewshot_file_dict
from dataset_conv import FtDataset as SFTDataset
# from dataset import FtDataset as SFTDataset
# from dataset import get_processed_dataset


def build_train_valid_test_datasets(data_prefix, data_impl, splits_string,
                                    train_valid_test_num_samples,
                                    seq_length, seed, skip_warmup,
                                    train_data_prefix=None,
                                    valid_data_prefix=None,
                                    test_data_prefix=None,
                                    return_doc_ids=False,
                                    fewshot_files=None,
                                    max_num_shot=5,
                                    fixed_shot=False,
                                    model_name=None):
    """Build train, valid, and test datasets."""

    if fewshot_files:
        fewshot_file_dict = get_fewshot_file_dict(fewshot_files)
    else:
        fewshot_file_dict = None

    if data_prefix:
        print_rank_0("Single data path provided for train, valid & test")

        # Single dataset.
        if len(data_prefix) == 1:
            if fewshot_file_dict:
                fewshot_input_file = fewshot_file_dict[data_prefix[0]]
            else:
                fewshot_input_file = None
            return _build_train_valid_test_datasets(data_prefix[0],
                                                    data_impl, splits_string,
                                                    train_valid_test_num_samples,
                                                    seq_length, seed, skip_warmup,
                                                    fewshot_input_file=fewshot_input_file,
                                                    max_num_shot=max_num_shot,
                                                    fixed_shot=fixed_shot,
                                                    model_name=model_name)

        # Blending dataset.
        # Parse the values.
        output = get_datasets_weights_and_num_samples(data_prefix,
                                                      train_valid_test_num_samples)
        prefixes, weights, datasets_train_valid_test_num_samples = output

        # Build individual datasets.
        train_datasets = []
        valid_datasets = []
        test_datasets = []
        for i in range(len(prefixes)):
            if fewshot_file_dict:
                fewshot_input_file = fewshot_file_dict[prefixes[i]]
            else:
                fewshot_input_file = None
            train_ds, valid_ds, test_ds = _build_train_valid_test_datasets(
                prefixes[i], data_impl, splits_string,
                datasets_train_valid_test_num_samples[i],
                seq_length, seed, skip_warmup,
                return_doc_ids, fewshot_input_file=fewshot_input_file,
                max_num_shot=max_num_shot, fixed_shot=fixed_shot,
                model_name=model_name)
            if train_ds:
                train_datasets.append(train_ds)
            if valid_ds:
                valid_datasets.append(valid_ds)
            if test_ds:
                test_datasets.append(test_ds)

        # Blend.
        blending_train_dataset = None
        if train_datasets:
            blending_train_dataset = BlendableDataset(train_datasets, weights)
        blending_valid_dataset = None
        if valid_datasets:
            blending_valid_dataset = BlendableDataset(valid_datasets, weights)
        blending_test_dataset = None
        if test_datasets:
            blending_test_dataset = BlendableDataset(test_datasets, weights)

        return (blending_train_dataset, blending_valid_dataset,
                blending_test_dataset)

    else:
        print_rank_0("Separate data paths provided for train, valid & test. Split string will be ignored.")

        train_dataset, valid_dataset, test_dataset = None, None, None
        # Single dataset.
        if train_data_prefix is not None:
            train_dataset = build_dataset("train", train_data_prefix, data_impl,
                                          train_valid_test_num_samples[0],
                                          seq_length, seed, skip_warmup)

        if valid_data_prefix is not None:
            valid_dataset = build_dataset("valid", valid_data_prefix, data_impl,
                                          train_valid_test_num_samples[1],
                                          seq_length, seed, False)

        if test_data_prefix is not None:
            test_dataset = build_dataset("test", test_data_prefix, data_impl,
                                         train_valid_test_num_samples[2],
                                         seq_length, seed, False)

        return (train_dataset, valid_dataset, test_dataset)


def _build_train_valid_test_datasets(data_prefix, data_impl, splits_string,
                                     train_valid_test_num_samples,
                                     seq_length, seed, skip_warmup,
                                     return_doc_ids=False, 
                                     fewshot_input_file=None, max_num_shot=5,
                                     fixed_shot=False,
                                     model_name=None):
    """Build train, valid, and test datasets using existing split"""

    args = get_args()

    # get fewshot samples
    if fewshot_input_file:
        fewshot_list = load_incontext_fewshot_samples(fewshot_input_file, max_num_shot)
    else:
        fewshot_list = None

    # Indexed dataset.
    indexed_dataset = get_processed_dataset(data_prefix, args.data_folder)

    train_dataset = SFTDataset(data_prefix, indexed_dataset["train"], seq_length, fewshot_list=fewshot_list, fixed_shot=fixed_shot, model_name=model_name)
    valid_dataset = SFTDataset(data_prefix, indexed_dataset["valid"], seq_length, fewshot_list=fewshot_list, fixed_shot=fixed_shot, model_name=model_name)
    test_dataset = SFTDataset(data_prefix, indexed_dataset["test"], seq_length, fewshot_list=fewshot_list, fixed_shot=fixed_shot, model_name=model_name)
    return (train_dataset, valid_dataset, test_dataset)


def build_dataset(dataset_name, data_prefix, data_impl, num_samples,
                  seq_length, seed, skip_warmup):
    dataset = None
    if len(data_prefix) == 1:
        dataset = _build_dataset(dataset_name,
                        data_prefix[0], data_impl,
                        num_samples, seq_length,
                        seed, skip_warmup)
    else:
        # Blending dataset.
        # Parse the values.
        output = get_datasets_weights_and_num_samples(data_prefix, num_samples)
        prefixes, weights, dataset_num_samples = output

        # Build individual datasets.
        datasets = []
        for i in range(len(prefixes)):
            ds = _build_dataset(dataset_name, prefixes[i],
                            data_impl, dataset_num_samples[i],
                            seq_length, seed, skip_warmup)
            if ds:
                datasets.append(ds)

        if datasets:
            dataset = BlendableDataset(datasets, weights)

    return dataset


def _build_dataset(dataset_name, data_prefix, data_impl,
                   num_samples, seq_length, seed, skip_warmup):
    """
    Build dataset. This method is called when individual
    train, valid, test datasets are provided
    """

    args = get_args()
    # Indexed dataset.
    indexed_dataset = get_processed_dataset(data_prefix, args.data_folder)

    dataset = SFTDataset(data_prefix, indexed_dataset[dataset_name], seq_length)

    return dataset


