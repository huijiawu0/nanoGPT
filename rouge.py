# -*- coding: utf-8 -*-

from bert_score.scorer import BERTScorer

bs = BERTScorer(model_type="bert-base-chinese", device="cuda")


def rouge_n(evaluated_sentences, reference_sentences, n=2):
    # 默认rouge_2
    if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
        return 0
    
    evaluated_ngrams = get_ngrams(n, evaluated_sentences)
    reference_ngrams = get_ngrams(n, reference_sentences)
    reference_ngrams_count = len(reference_ngrams)
    if reference_ngrams_count == 0:
        return 0
    
    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_ngrams_count = len(overlapping_ngrams)
    return overlapping_ngrams_count / reference_ngrams_count


def rouge_1(evaluated_sentences, reference_sentences):
    evaluated_sentences = ''.join(evaluated_sentences)
    reference_sentences = ''.join(reference_sentences)
    return rouge_n(evaluated_sentences, reference_sentences, n=1)


def rouge_2(evaluated_sentences, reference_sentences):
    evaluated_sentences = ''.join(evaluated_sentences)
    reference_sentences = ''.join(reference_sentences)
    return rouge_n(evaluated_sentences, reference_sentences, n=2)


def F_1(evaluated_sentences, reference_sentences, beta=1):
    evaluated_sentences = ''.join(evaluated_sentences)
    reference_sentences = ''.join(reference_sentences)
    if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
        return 0
    
    evaluated_ngrams = get_ngrams(beta, evaluated_sentences)  # equal to retrieved set
    reference_ngrams = get_ngrams(beta, reference_sentences)  # equal to relevant set
    evaluated_ngrams_num = len(evaluated_ngrams)
    reference_ngrams_num = len(reference_ngrams)
    
    if reference_ngrams_num == 0 or evaluated_ngrams_num == 0:
        return 0
    
    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_ngrams_num = len(overlapping_ngrams)
    if overlapping_ngrams_num == 0:
        return 0
    return 2 * overlapping_ngrams_num / (reference_ngrams_num + evaluated_ngrams_num)


def get_ngrams(n, text):
    ngram_set = set()
    text_length = len(text)
    max_index_ngram = text_length - n
    
    for i in range(max_index_ngram + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


def to_string(bytes_or_str):
    """receive str or unicode and always return string"""
    if isinstance(bytes_or_str, bytes):
        value = bytes_or_str.decode("utf-8")
    else:
        value = bytes_or_str
    return value


if __name__ == "__main__":
    rg1, rg2 = [], []
    bs_list = []
    import sys
    
    with open(sys.argv[1], 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            _, ref, sys = line.split("|||")
            multi_sys = sys.split("<s>")
            temp1 = [F_1(ele, ref, beta=1) for ele in multi_sys]
            temp2 = [F_1(ele, ref, beta=2) for ele in multi_sys]
            sys_outs = [[ele] for ele in multi_sys]
            score = bs.score([ref], sys_outs)[-1].tolist()[0]
            rg1.append(max(temp1))
            rg2.append(max(temp2))
            bs_list.append(score)
    
    print("ROUGE1:", sum(rg1) / len(rg1))
    print("ROUGE2:", sum(rg2) / len(rg2))
    print("bertscore:", sum(bs_list) / len(bs_list))

