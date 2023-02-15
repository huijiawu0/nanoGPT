from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import sys

hf_model_path = sys.argv[1]
tokenizer = GPT2Tokenizer.from_pretrained(hf_model_path)
config = GPT2Config.from_pretrained(hf_model_path)
model = GPT2LMHeadModel.from_pretrained(hf_model_path)

model.to("cuda")


def get_single(question):
    inputs = tokenizer(question, return_tensors='pt').to("cuda")
    assert isinstance(model, GPT2LMHeadModel)
    # stop_tok = tokenizer.encode('。')[0]
    generation_output = model.generate(**inputs,
                                       do_sample=False,
                                       early_stopping=True,
                                       max_length=500,
                                       eos_token_id=50256,
                                       pad_token_id=0,
                                       num_return_sequences=1
                                       )
    ret = []
    for idx, sentence in enumerate(generation_output):
        s = tokenizer.decode(sentence).split('<|endoftext|>')[0]
        ret.append(s)
    return ret

out = sys.argv[2]
with open("qa_pair.txt", 'r') as f, open(out, 'w') as g:
    for idx, line in enumerate(f):
        r = line.strip().split('|||')
        if len(r) != 2:
            continue
        q, a = r
        res = get_single(q)
        g.write("%s|||%s|||%s\n" % (q, a, '<s>'.join(res)))
        if idx > 100:
            break