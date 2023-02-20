import os
import requests
import tiktoken
import numpy as np
import json
from datasets import load_dataset
from tqdm import tqdm
import s3fs
storage_options = {"key": "AKIA5AKOSQ7KIVCK7CJZ", "secret": "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"}
s3 = s3fs.S3FileSystem(**storage_options)

from transformers import GPT2TokenizerFast

num_proc = os.cpu_count()
print(num_proc)
enc = tiktoken.get_encoding("gpt2")


def process(example):
    title = example['title'].strip()
    ids = enc.encode_ordinary(title)
    ids.append(enc.eot_token)
    content = example['content'].strip()
    ids.extend(enc.encode_ordinary(content))
    ids.append(enc.eot_token)
    out = {'ids': ids, 'len': len(ids)}
    return out


def run_single(in_f, out_f, dfile):
    dataset = load_dataset('json', data_files={'train': os.path.join(in_f, dfile)}, num_proc=num_proc, keep_in_memory=True)
    # split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    # split_dataset['val'] = split_dataset.pop('test')  # rename the test split to val
    #
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
        dname = dfile.split('.json')[0]
        binname = f'{dname}_{split}.bin'
        filename = os.path.join(out_f, binname)
        dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        
        print(f"writing {filename}...")
        idx = 0
        for example in tqdm(dset):
            arr[idx: idx + example['len']] = example['ids']
            idx += example['len']
            # print("\tidx: %d/%d" % (idx, arr_len))
        arr.flush()
        print(f"put file {filename} to s3...")
        s3.put_file(filename, 's3://datawd/%s' % binname)
        os.popen("rm %s" % filename)
        # print(f"put file {filename} to s3 end")


import os, sys
in_folder1 = sys.argv[1]
out_f = '.'

existing = set(b.strip().split('_')[0] + '.json' for b in os.listdir(out_f))
for idx, fi in tqdm(enumerate(os.listdir(in_folder1))):
    if fi in existing:
        print("file %s has been already converted" % fi)
        continue
    else:
        if fi.startswith('part-'):
            # print("processing %s, %d/366" % (fi, idx))
            run_single(in_folder1, out_f, fi)
