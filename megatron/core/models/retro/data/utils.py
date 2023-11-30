# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

from collections import defaultdict
import glob
import numpy as np
import os
import torch
from tqdm import tqdm
from types import SimpleNamespace
from typing import Callable

from megatron.core import parallel_state
from megatron.core.datasets.blended_megatron_dataset_config import GPTDatasetConfig

from .config import RetroPreprocessingConfig
from .external_libs import h5py


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def extract_data_config(config):
    return config.retro_gpt_datasets.train[0].config


def get_config_path(project_dir):
    '''Config copy stored within retro project dir.'''
    return os.path.join(project_dir, "config.json")


def get_num_chunks_per_sample(sample_length, chunk_length):
    '''Compute seq_length // chunk_length.'''
    assert sample_length % chunk_length == 0
    return sample_length // chunk_length


def get_gpt_data_dir(project_dir):
    return os.path.join(project_dir, "data")


def core_gpt_dataset_config_from_retro_preprocessing_config(
    config: RetroPreprocessingConfig,
    is_dataset_built_on_rank: bool,
) -> GPTDatasetConfig:
    data_dir = get_gpt_data_dir(config.retro_project_dir)
    blend = list(config.retro_gpt_data_path)
    for i in range(len(blend) - 1, -1, -2):
        blend[i] = os.path.join(data_dir, blend[i])
    return GPTDatasetConfig(
        is_built_on_rank=is_dataset_built_on_rank,
        random_seed=config.retro_gpt_seed,
        sequence_length=config.retro_gpt_seq_length,
        blend=blend,
        split=config.retro_gpt_split,
        path_to_cache=config.retro_gpt_data_cache_path,
        return_document_ids=True,
    )


class GPTToTextDataset(torch.utils.data.Dataset):
    '''Dataset to convert GPT tokens to text.'''

    def __init__(self, gpt_dataset, gpt_tokenizer):

        super().__init__()

        self.gpt_dataset = gpt_dataset
        self.gpt_tokenizer = gpt_tokenizer

    def __len__(self):
        return len(self.gpt_dataset)

    def __getitem__(self, idx):
        gpt_token_ids = self.gpt_dataset[idx]["text"].tolist()
        text = self.gpt_tokenizer.detokenize(gpt_token_ids)
        return {"text": text}


def get_blocks(
    project_dir: str,
    n_samples: int,
    block_size: int,
    validate: Callable = None,
):
    '''Divide range [0, num_samples) to sequence of block ranges.

    This is a core method within the concept of block processing. The idea
    is to divide a range (size n_samples) into a sequence of blocks. Each
    block corresponds to a file within 'project_dir' with name
    '{start_idx}-{end_idx}.hdf5'. This method checks for the existence of
    these files, and returns two lists, one for existing blocks and one for
    missing blocks.
    '''

    # Block ranges.
    block_start_idxs = list(range(0, n_samples, block_size))
    block_end_idxs = [ min(n_samples, i + block_size) for i in block_start_idxs ]
    block_ranges = list(zip(block_start_idxs, block_end_idxs))

    # All block files (existing + missing).
    n_digits = int(np.ceil(np.log(n_samples) / np.log(10)) + 1)
    all_blocks = [{
        "range" : r,
        "path" : os.path.join(
            project_dir,
            "%s-%s.hdf5" % tuple([ str(i).zfill(n_digits) for i in r ]),
        )
    } for r in block_ranges]
    all_block_path_set = set(block["path"] for block in all_blocks)

    # Validate function.
    validate = (lambda f : None) if validate is None else validate

    # Delete corrupt files.
    if torch.distributed.get_rank() == 0:
        existing_block_paths = [block["path"]
                                for block in all_blocks
                                if os.path.exists(block["path"])]
        for index, path in enumerate(
                tqdm(existing_block_paths, "validating block.")):

            assert path in all_block_path_set, "unexpected filename, '%s'." % path

            try:
                f = h5py.File(path, "r")
            except:
                os.remove(path)
                continue

            try:
                validate(f)
            except:
                os.remove(path)
            finally:
                f.close()

    # Wait for files to be deleted.
    torch.distributed.barrier()

    # Collect blocks.
    blocks = SimpleNamespace(
        existing=[ b for b in all_blocks if os.path.exists(b["path"]) ],
        missing=[ b for b in all_blocks if not os.path.exists(b["path"]) ],
    )

    return blocks


def get_blocks_by_rank(
    project_dir: str,
    n_samples: int,
    block_size: int,
    validate: Callable = None,
):
    '''Divide existing and missing blocks evenly across all ranks.

    See 'get_blocks()' above for description. The returned lists of existing and
    missing blocks are split evenly across ranks via interleaving. This way,
    each rank has a roughly equal number of blocks to process for a
    downstream operation.
    '''

    # Get world blocks.
    blocks = get_blocks(project_dir, n_samples, block_size, validate)

    # This rank's existing and missing files.
    data_parallel_rank = parallel_state.get_data_parallel_rank()
    data_parallel_world_size = parallel_state.get_data_parallel_world_size()
    rank_existing_blocks = blocks.existing[data_parallel_rank:len(blocks.existing):data_parallel_world_size]
    rank_missing_blocks = blocks.missing[data_parallel_rank:len(blocks.missing):data_parallel_world_size]

    # Extend rank's existing and missing blocks (with None) such that all ranks
    # have equal length lists. This allows for easier tracking of global progress.
    def get_world_max(n):
        n_tensor = torch.cuda.LongTensor([n])
        torch.distributed.all_reduce(n_tensor, op=torch.distributed.ReduceOp.MAX)
        return n_tensor.item()

    max_n_existing = get_world_max(len(rank_existing_blocks))
    max_n_missing = get_world_max(len(rank_missing_blocks))

    rank_existing_blocks += [None] * (max_n_existing - len(rank_existing_blocks))
    rank_missing_blocks += [None] * (max_n_missing - len(rank_missing_blocks))

    # Collect blocks.
    blocks = SimpleNamespace(
        n_existing_world = len(blocks.existing),
        n_missing_world = len(blocks.missing),
        existing = rank_existing_blocks,
        missing = rank_missing_blocks,
    )

    return blocks


def get_sampled_blocks_by_rank(
    project_dir: str,
    n_samples: int,
    block_size: int,
    validate: Callable = None,
    fraction: float = None,
):
    '''Sample existing and missing blocks evenly across all ranks.

    See 'get_blocks_by_rank()' above for description. The returned lists of
    blocks are randomly sampled (without replacement) to yield
    `fraction * len(blocks)` number of blocks.
    '''

    # Get blocks.
    blocks = get_blocks_by_rank(project_dir, n_samples, block_size, validate)

    # Randomly sample blocks.
    def sample_blocks(_blocks):
        n_blocks_sample = int(np.ceil(fraction * len(_blocks)))
        sampled_blocks = [ b for b in _blocks if b is not None ]

        np.random.seed(None)
        np.random.shuffle(sampled_blocks)

        sampled_blocks = sampled_blocks[:n_blocks_sample]
        sampled_blocks += [None] * (n_blocks_sample - len(sampled_blocks))

        return sampled_blocks

    blocks.existing = sample_blocks(blocks.existing)
    blocks.missing = sample_blocks(blocks.missing)

    return blocks


class BlockPathMap:
    '''Map an index to its containing block path.

    The common use for this class is to have a directory of files containing
    blocks of processed data, of uniform block size (e.g., 100k samples per
    file). Each file must follow a naming convention of 'startIdx-endIdx.[ext]',
    where 'endIdx' minus 'startIdx' must equal the block size, with the possible
    exception of the final block. Given an input index, this class maps the
    index to the containing block file.
    '''

    @classmethod
    def from_dir(cls, dir, block_size, ext="hdf5"):
        '''Get list of block files, and create map.'''
        assert os.path.isdir(dir), f"directory not found, '{dir}'."
        return cls(sorted(glob.glob(dir + f"/*.{ext}")), block_size)

    def __init__(self, block_paths, block_size):
        self.max_idx = 0
        self.block_path_map = {}
        for block_path in block_paths:
            name = os.path.splitext(os.path.basename(block_path))[0]
            start_idx, end_idx = [ int(i) for i in name.split("-") ]
            self.block_path_map[start_idx] = block_path
            self.max_idx = max(self.max_idx, end_idx)
        self.block_size = block_size

    def __str__(self):
        return "%d paths" % len(self.block_path_map)

    def __getitem__(self, idx):
        '''Get block path from index.'''
        block_start_idx = self.block_size * (idx // self.block_size)
        block_path = self.block_path_map[block_start_idx]
        return block_path
