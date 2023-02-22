import tiktoken
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

num_proc = 6
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
    dataset = load_dataset('json', data_files={'train': os.path.join(in_f, dfile)}, num_proc=num_proc, keep_in_memory=False)
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
        arr.flush()
        

import os

in_folder1 = 'data'
out_f = 'bin'

existing = set(b.strip().split('_')[0] + '.json' for b in os.listdir(out_f))
for idx, fi in tqdm(enumerate(os.listdir(in_folder1))):
    if fi in existing:
        print("file %s has been already converted" % fi)
        continue
    else:
        if fi.startswith('part-'):
            run_single(in_folder1, out_f, fi)
