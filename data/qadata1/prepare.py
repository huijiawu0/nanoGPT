import tiktoken
from datasets import load_dataset, DatasetDict, Dataset  # huggingface datasets
import numpy as np
import os

from sklearn.model_selection import train_test_split
from tqdm import tqdm


num_proc = 8

dataset = []
with open('qa_pair.txt', 'r') as f:
    for line in f:
        rline = line.strip().split('|||')
        if len(rline) == 2:
            a, b = rline
            dataset.append(a + b)

train_ds, valid_ds = train_test_split(dataset, test_size=0.05)
split_dataset = DatasetDict({'train': Dataset.from_dict({"text": train_ds}),
                             'val': Dataset.from_dict({"text": valid_ds})})

enc = tiktoken.get_encoding("gpt2")


def process(example):
    ids = enc.encode_ordinary(' '.join(example['text']))  # encode_ordinary ignores any special tokens
    ids.append(enc.eot_token)  # add the end of text token, e.g. 50256 for gpt2 bpe
    # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
    out = {'ids': ids, 'len': len(ids)}
    return out


# tokenize the dataset
tokenized = split_dataset.map(
    process,
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
