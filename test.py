import json
import numpy as np
from matplotlib import pyplot as plt
from nltk.tokenize import word_tokenize
from sklearn.metrics import f1_score, accuracy_score, ConfusionMatrixDisplay
import gensim.downloader as api
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from config import model_file_path, word_embedding, test_file_path
from Utils import dataReader,prepare_data_for_model

def two_way_from_three_way(labels):
    two_way=[]
    for label in labels:
        if label!='entailment':
            two_way.append('non-entailment')
        else:
            two_way.append('entailment')
    return two_way

def model_res_to_labels(results):
    convert_table = {
        0: "entailment",
        1: "neutral",
        2: "contradiction"
    }
    labels=[]
    for res in results:
        index=np.argmax(res)
        labels.append(convert_table[index])
    return labels

def test():
    glove = api.load(word_embedding)
    max_length = 50

    model = load_model(model_file_path)
    print("loaded glove")
    data_for_test = dataReader(test_file_path)
    test_sentences,gold_labels = prepare_data_for_model(data_for_test,glove)
    test_padded = np.array(pad_sequences(test_sentences, maxlen=max_length, padding='post', truncating='post'))
    predictions = model.predict(test_padded)
    predictions_labels=model_res_to_labels(predictions)

    accuracy_three_way = accuracy_score(gold_labels, predictions_labels)
    ConfusionMatrixDisplay.from_predictions(
        gold_labels, predictions_labels)
    plt.show()
    f1_three_way=f1_score(gold_labels,predictions_labels)

    two_way_predictions=two_way_from_three_way(predictions_labels)
    gold_labels_predictions=two_way_from_three_way(gold_labels)
    accuracy_two_way = accuracy_score(gold_labels_predictions, two_way_predictions)
    ConfusionMatrixDisplay.from_predictions(
        gold_labels_predictions, two_way_predictions)
    plt.show()
    f1_two_way=f1_score(gold_labels_predictions,two_way_predictions)

    print(f"""
            Two way results:
                Accuracy={accuracy_two_way}
                f1:
                    entailment={f1_two_way[0]}
                    non-entailment={f1_two_way[1]}
            Three way results:
                Accuracy={accuracy_three_way}
                f1:
                    entailment={f1_three_way[0]}
                    neutral={f1_three_way[1]}
                    contradiction={f1_three_way[2]}""")
    return

if __name__ == '__main__':
    test()