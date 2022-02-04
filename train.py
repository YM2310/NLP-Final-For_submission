import json
import numpy as np
from nltk.tokenize import word_tokenize
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gensim.downloader as api
from config import word_embedding
from config import train_dataset_file_path
from Utils import prepare_data_for_model,dataReader

def labels_to_ints(labels):
    convert_table = {
        "entailment": 0,
        "neutral": 1,
        "contradiction": 2
    }
    return [convert_table[label] for label in labels]


def create_pretrained_embedding_layer(glove):
    vocab_len = len(glove)
    emb_dim = glove["man"].shape[0]
    embeddingMatrix = np.zeros((vocab_len, emb_dim))
    for word in glove.index_to_key:
        embeddingMatrix[glove.key_to_index[word], :] = glove[word]
    embeddingLayer = layers.Embedding(vocab_len, emb_dim, weights=[embeddingMatrix], trainable=False)
    return embeddingLayer


def chunks(l, n):
    d, r = divmod(len(l), n)
    for i in range(n):
        si = (d + 1) * (i if i < r else r) + d * (0 if i < r else i - r)
        yield l[si:si + (d + 1 if i < r else d)]

def create_and_compile_model(word_embbeding):
    pretrainedEmbeddingLayer = create_pretrained_embedding_layer(word_embbeding)
    model = keras.models.Sequential()
    model.add(pretrainedEmbeddingLayer)
    model.add(layers.Bidirectional(layers.LSTM(128, dropout=0.3)))
    model.add(layers.Dense(3, activation='softmax'))
    loss = keras.losses.CategoricalCrossentropy(from_logits=False)
    optim = keras.optimizers.Adam(learning_rate=0.001)
    metrics = ["accuracy"]
    model.summary()
    model.compile(loss=loss, optimizer=optim, metrics=metrics)
    return model

def train_lstm_model():
    glove = api.load(word_embedding)
    print("loaded glove")
    data_for_train = dataReader(train_dataset_file_path)
    print('data ready')
    k_fold_num = 8
    max_length = 50
    num_of_labels=3
    sentences, gold_labels = prepare_data_for_model(data_for_train, glove)
    model=create_and_compile_model(glove)
    sentences_chunks = list(chunks(sentences, k_fold_num))
    labels_chunks = list(chunks(gold_labels, k_fold_num))

    for i in range(k_fold_num):
        train_sequences = []
        train_labels = []
        for j in range(k_fold_num):
            if j != i:
                train_sequences.extend(sentences_chunks[j])
                train_labels.extend(labels_chunks[j])
        train_labels = np.array(labels_to_ints(train_labels))

        validation_sequences = sentences_chunks[i]
        validation_labels = np.array(labels_to_ints(labels_chunks[i]))

        train_padded = np.array(pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post'))
        train_labels = keras.utils.to_categorical(train_labels, num_classes=num_of_labels)

        validation_padded = np.array(
            pad_sequences(validation_sequences, maxlen=max_length, padding='post', truncating='post'))
        validation_labels = keras.utils.to_categorical(validation_labels, num_classes=num_of_labels)

        model.fit(train_padded, train_labels, epochs=10, validation_data=(validation_padded, validation_labels),
                  verbose=2)
        print('training iteration: ', i)
    model.save('LSTM-Model.h5')


if __name__ == '__main__':
    train_lstm_model()
