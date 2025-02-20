import os
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from cnn_model import NewsCNN


class ModelTrainer:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.label_encoder = None
        self.model = None

    def train(self, data_path, model_save_path):
        df = pd.read_csv(data_path)
        texts = df['text'].values
        labels = df['label'].values

        X = np.array([self.preprocessor.text_to_vectors(text)
                     for text in texts])

        self.label_encoder = {label: idx for idx,
                              label in enumerate(np.unique(labels))}
        y = to_categorical([self.label_encoder[label] for label in labels])

        self.model = NewsCNN(input_shape=(self.preprocessor.max_seq_length,
                                          self.preprocessor.embedding_dim),
                             num_classes=len(self.label_encoder)).build_model()

        early_stop = EarlyStopping(monitor='val_loss', patience=3)
        self.model.fit(X, y,
                       batch_size=32,
                       epochs=20,
                       validation_split=0.2,
                       callbacks=[early_stop])

        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        self.model.save(model_save_path)
        np.save(f"{model_save_path}_label_encoder.npy", self.label_encoder)
