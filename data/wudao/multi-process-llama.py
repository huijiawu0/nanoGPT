import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
import os
import transformers
import multiprocessing

num_proc = round(os.cpu_count() * 0.6)
# num_proc = 1
print(num_proc)
assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaForCausalLM, LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained(
    "decapoda-research/llama-7b-hf", add_eos_token=True
)
tokenizer.pad_token_id = 0
enc = tiktoken.get_encoding("gpt2")

# CUTOFF_LEN = 2048


def process(example):
    title = example['title'].strip()
    content = example['content'].strip()
    ids = tokenizer(
                title + content,
                truncation=False
            )["input_ids"]
    out = {'ids': ids, 'len': len(ids)}
    return out


def run_single(in_f, out_f, dfile):
    dataset = load_dataset('json', data_files={'train': os.path.join(in_f, dfile)}, num_proc=1,
                           keep_in_memory=False, cache_dir='.')
    tokenized = dataset.shuffle().map(
        process,
        num_proc=1,
        keep_in_memory=True
    )
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'])
        print("arr_len: %d" % arr_len)
        dname = dfile.split('.json')[0]
        binname = f'{dname}_{split}.bin'
        filename = os.path.join(out_f, binname)
        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        
        print(f"writing {filename}...")
        idx = 0
        for example in tqdm(dset):
            arr[idx: idx + example['len']] = example['ids']
            idx += example['len']
        arr.flush()


if __name__ == '__main__':
    import sys
    in_folder1 = sys.argv[1]
    out_f = sys.argv[2]
    
    existing = set(b.strip().split('_')[0] + '.json' for b in os.listdir(out_f))
    pool = multiprocessing.Pool(processes=num_proc)
    
    results = []
    for idx, fi in enumerate(os.listdir(in_folder1)):
        if fi in existing:
            print("file %s has been already converted" % fi)
            continue
        else:
            print("start processing file ", fi)
            if fi.startswith('part-'):
                result = pool.apply_async(run_single, args=(in_folder1, out_f, fi))
                results.append(result)
    
    for result in results:
        result.get()
    
    pool.close()
    pool.join()
