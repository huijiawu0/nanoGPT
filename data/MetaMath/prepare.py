# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
import json
import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset  # huggingface datasets
from datasets import Dataset, DatasetDict


# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def get_examples2(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for ex in data:
        ex.update(query=ex["query"] + "\n")
    
    print(f"{path} has {len(data)} examples")
    return data


def get_examples(path):
    examples = read_jsonl(path)
    for ex in examples:
        ex.update(query=ex["query"] + "\n")

    print(f"{path} has {len(examples)} examples")
    return examples


if __name__ == '__main__':
    enc = tiktoken.get_encoding("gpt2")
    train_examples = get_examples2("MetaMath-40K.json")
    eval_examples = get_examples("GSM8K_test.jsonl")
    train_dataset = Dataset.from_dict({k: [dic[k] for dic in train_examples] for k in train_examples[0]})
    eval_dataset = Dataset.from_dict({k: [dic[k] for dic in eval_examples] for k in eval_examples[0]})
    split_dataset = DatasetDict({"train": train_dataset, "val": eval_dataset})

    def process(example):
        ids = enc.encode_ordinary(example['query'] + example['response'])  # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token)  # add the end of text token, e.g. 50256 for gpt2 bpe
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        out = {'ids': ids, 'len': len(ids)}
        return out

    # tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=['query', 'response'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )
    
    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024
        
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
    
    # train.bin is ~17GB, val.bin ~8.5MB
    # train has ~9B tokens (9,035,582,198)
    # val has ~4M tokens (4,434,897)
    
    # to read the bin files later, e.g. with numpy:
    # m = np.memmap('train.bin', dtype=np.uint16, mode='r')
