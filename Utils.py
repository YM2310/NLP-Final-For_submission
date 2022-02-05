import json

from nltk.tokenize import word_tokenize


def dataReader(file_name):
    with open(file_name, 'r') as corpus_file:
        corpus = [json.loads(jline) for jline in corpus_file.read().splitlines()]
    return corpus

def sentence2sequence(sentence,glove_model):
    sentence=sentence.lower()
    tokens = word_tokenize(sentence)
    sequence=[]
    for token in tokens:
        if token in glove_model.key_to_index:
            sequence.append(glove_model.key_to_index[token])
        else:
            sequence.append(0)
    return sequence

def prepare_data_for_model(data,glove):
    gold_labels = []
    sentences = []
    for line in data:
        if line["gold_label"] == '-':
            continue
        sentences.append(sentence2sequence(f"{line['sentence1']} {line['sentence2']}",glove))
        gold_labels.append(line['gold_label'])
    return sentences, gold_labels