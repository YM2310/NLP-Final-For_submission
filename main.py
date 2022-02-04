import json
import numpy as np
from matplotlib import pyplot as plt
from nltk.tokenize import word_tokenize
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import gensim.downloader as api
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from config import model_download_url
from config import model_file_path
from config import word_embedding
import test

def obtain_model_from_url(model_download_url):
    model_file_path='new path'
    print("model_path")





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    model_path=obtain_model_from_url(model_download_url)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
