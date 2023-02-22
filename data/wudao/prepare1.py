import os
import requests
import tiktoken
import numpy as np
import json
from datasets import load_dataset
from tqdm import tqdm
# import s3fs
# storage_options = {"key": "AKIA5AKOSQ7KIVCK7CJZ", "secret": "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"}
# s3 = s3fs.S3FileSystem(**storage_options)

from transformers import GPT2TokenizerFast

num_proc = os.cpu_count()
print(num_proc)
enc = tiktoken.get_encoding("gpt2")

import sys
dfiles = sys.argv[1]
print(dfiles)
dataset = load_dataset('json', data_files={'train': dfiles}, num_proc=num_proc, keep_in_memory=True)
# split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
# split_dataset['val'] = split_dataset.pop('test')  # rename the test split to val


def process(example):
    title = example['title'].strip()
    ids = enc.encode_ordinary(title)
    ids.append(enc.eot_token)
    content = example['content'].strip()
    ids.extend(enc.encode_ordinary(content))
    ids.append(enc.eot_token)
    out = {'ids': ids, 'len': len(ids)}
    return out


tokenized = dataset.map(
    process,
    remove_columns=['title', 'content'],
    desc="tokenizing the splits",
    num_proc=num_proc,
    keep_in_memory=True
)

for split, dset in tokenized.items():
    arr_len = np.sum(dset['len'])
    print("arr_len: %d" % arr_len)
    dname = 'train'
    binname = f'{dname}.bin'
    # filename = os.path.join(out_f, binname)
    dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
    arr = np.memmap(binname, dtype=dtype, mode='w+', shape=(arr_len,))
    
    print(f"writing {binname}...")
    idx = 0
    for example in tqdm(dset):
        arr[idx: idx + example['len']] = example['ids']
        idx += example['len']
        # print("\tidx: %d/%d" % (idx, arr_len))
    arr.flush()