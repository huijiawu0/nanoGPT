from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import sys

hf_model_path = sys.argv[1]
tokenizer = GPT2Tokenizer.from_pretrained(hf_model_path)
config = GPT2Config.from_pretrained(hf_model_path)
model = GPT2LMHeadModel.from_pretrained(hf_model_path)

model.to("cuda")
max_length = int(sys.argv[2])
num_return_sequences = int(sys.argv[3])


def get_single(question):
    # question = "写一篇神经网络的报告"
    inputs = tokenizer(question, return_tensors='pt').to("cuda")
    assert isinstance(model, GPT2LMHeadModel)
    generation_output = model.generate(**inputs,
                                       return_dict_in_generate=True,
                                       output_scores=True,
                                       max_length=max_length,
                                       do_sample=True,
                                       top_p=0.6,
                                       # num_beams=5,
                                       eos_token_id=50256,
                                       pad_token_id=0,
                                       num_return_sequences=num_return_sequences,
                                       )
    ret = []
    for idx, sentence in enumerate(generation_output.sequences):
        s = tokenizer.decode(sentence).split('<|endoftext|>')[0]
        ret.append(s)
    return ret

out = sys.argv[4]
with open("qa_pair.txt", 'r') as f, open(out, 'w') as g:
    for idx, line in enumerate(f):
        r = line.strip().split('|||')
        if len(r) != 2: continue
        q, a = r
        res = get_single(q)
        g.write("%s|||%s|||%s\n" % (q, a, '<s>'.join(res)))
        if idx > 10:
            break