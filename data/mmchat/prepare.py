import tiktoken
from datasets import load_dataset, DatasetDict  # huggingface datasets
import numpy as np
import os
from tqdm import tqdm


# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8

dataset = load_dataset("silver/mmchat")

split_dataset = DatasetDict({'train': dataset["train"], 'val': dataset['validation']})

enc = tiktoken.get_encoding("gpt2")


def process(example):
    ids = enc.encode_ordinary(' '.join(example['dialog']))  # encode_ordinary ignores any special tokens
    ids.append(enc.eot_token)  # add the end of text token, e.g. 50256 for gpt2 bpe
    # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
    out = {'ids': ids, 'len': len(ids)}
    return out


# tokenize the dataset
tokenized = split_dataset.map(
    process,
    remove_columns=['weibo_content', 'imgs'],
    desc="tokenizing the splits",
    num_proc=num_proc,
)


for split, dset in tokenized.items():
    arr_len = np.sum(dset['len'])
    filename = os.path.join(os.path.dirname('.'), f'{split}.bin')
    dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

    print(f"writing {filename}...")
    idx = 0
    for example in tqdm(dset):
        arr[idx : idx + example['len']] = example['ids']
        idx += example['len']
    arr.flush()
