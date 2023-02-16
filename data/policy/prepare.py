import os
import requests
import tiktoken
import numpy as np
import json
from datasets import load_dataset
from tqdm import tqdm

num_proc = 8

dataset = load_dataset('text', data_files={'train': 'policy_plain_text_100.txt'})

split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
split_dataset['val'] = split_dataset.pop('test')  # rename the test split to val


def process(example):
    ids = enc.encode_ordinary(example['text'])  # encode_ordinary ignores any special tokens
    ids.append(enc.eot_token)  # add the end of text token, e.g. 50256 for gpt2 bpe
    # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
    out = {'ids': ids, 'len': len(ids)}
    return out


enc = tiktoken.get_encoding("gpt2")

# tokenize the dataset
tokenized = split_dataset.map(
    process,
    remove_columns=['text'],
    desc="tokenizing the splits",
    num_proc=num_proc,
)

# concatenate all the ids in each dataset into one large file we can use for training
for split, dset in tokenized.items():
    arr_len = np.sum(dset['len'])
    filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
    dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    
    print(f"writing {filename}...")
    idx = 0
    for example in tqdm(dset):
        arr[idx: idx + example['len']] = example['ids']
        idx += example['len']
    arr.flush()
