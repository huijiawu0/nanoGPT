"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext

import tiktoken
import torch
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume'  # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out'  # ignored if init_from is not 'resume'
out_file = 'qa_pair_res.txt'
start = "\n"  # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10  # number of samples to draw
max_new_tokens = 500  # number of tokens generated in each sample
temperature = 0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200  # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16'  # 'float32' or 'bfloat16' or 'float16'
compile = False  # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read())  # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))
elif init_from == "to_hf":
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    # gptconf['vocab_size'] = 50257  # always 50257 for GPT model checkpoints
    # gptconf['block_size'] = 1024  # always 1024 for GPT model checkpoints
    # gptconf['bias'] = True  # always True for GPT model checkpoints
    model_pre = GPT(gptconf)
    sd = model_pre.state_dict()
    sd_keys = sd.keys()
    sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model_pre.load_state_dict(state_dict)

    conf = GPT2Config(vocab_size=gptconf.vocab_size, n_layer=gptconf.n_layer, n_head=gptconf.n_head, n_embd=gptconf.n_embd)
    model = GPT2LMHeadModel(conf)
    sd_hf = model.state_dict()
    sd_keys_hf = sd_hf.keys()
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
    assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
    for k in sd_keys_hf:
        if any(k.endswith(w) for w in transposed):
            # special treatment for the Conv1D weights we need to transpose
            assert sd_hf[k].shape[::-1] == sd[k].shape
            with torch.no_grad():
                sd_hf[k].copy_(sd[k].t())
        else:
            # vanilla copy over the other parameters
            if sd_hf[k].shape != sd[k].shape:
                raise Exception("hf %s vs. %s" % (sd_hf[k].shape, sd[k].shape))
            with torch.no_grad():
                sd_hf[k].copy_(sd[k])

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)  # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint[
    'config']:  # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
elif init_from == 'to_hf':
    tokenizer = GPT2Tokenizer.from_pretrained("IDEA-CCNL/Wenzhong-GPT2-110M")
    encode = lambda x: tokenizer(x, return_tensors='pt').to(device)
    decode = lambda x: tokenizer.decode(x).split('<|endoftext|>')[0]
else:
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)


def gen(start):
    start_ids = encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
    ret = []
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                ret.append(decode(y[0].tolist()))
    return ret


def gen_hf(question):
    input_ids = encode(question)
    assert isinstance(model, GPT2LMHeadModel)
    generation_output = model.generate(**input_ids,
                                       do_sample=False,
                                       early_stopping=True,
                                       max_length=500,
                                       eos_token_id=50256,
                                       pad_token_id=0,
                                       num_return_sequences=1
                                       )
    ret = []
    for idx, sentence in enumerate(generation_output):
        s = decode(sentence)
        ret.append(s)
    return ret


# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f, open(out_file, 'w') as g:
        for idx, line in enumerate(f):
            r = line.strip().split('|||')
            if len(r) != 2:
                continue
            q, a = r
            if init_from == "to_hf":
                res = gen_hf(q)
            else:
                res = gen(q)
            g.write("%s|||%s|||%s\n" % (q, a, '<s>'.join(res)))
            if idx > 100:
                break
