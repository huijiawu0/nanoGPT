"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""

# download the tiny shakespeare dataset
import tiktoken
from datasets import load_dataset, DatasetDict, Dataset
import numpy as np
import os

from tqdm import tqdm

num_proc = 8
dataset = load_dataset("wikipedia", language="zh", date="20230120", beam_runner="DirectRunner", ignore_verifications=True)

# owt by default only contains the 'train' split, so create a test split
split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
split_dataset['val'] = split_dataset.pop('test')  # rename the test split to val

# this results in:
# >>> split_dataset
# DatasetDict({
#     train: Dataset({
#         features: ['text'],
#         num_rows: 8009762
#     })
#     val: Dataset({
#         features: ['text'],
#         num_rows: 4007
#     })
# })

# we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
enc = tiktoken.get_encoding("gpt2")


def process(example):
    ids = enc.encode_ordinary(example['text'])  # encode_ordinary ignores any special tokens
    ids.append(enc.eot_token)  # add the end of text token, e.g. 50256 for gpt2 bpe
    # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
    out = {'ids': ids, 'len': len(ids)}
    return out


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


