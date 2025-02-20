from flask import Flask, request, jsonify
import numpy as np
from keras.models import load_model
from data_preprocessor import DataPreprocessor
from model_trainer import ModelTrainer

app = Flask(__name__)


class PredictionService:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.loaded_models = {}

    def load_model(self, model_path):
        if model_path not in self.loaded_models:
            model = load_model(model_path)
            label_encoder = np.load(
                f"{model_path}_label_encoder.npy", allow_pickle=True).item()
            self.loaded_models[model_path] = (model, label_encoder)
        return self.loaded_models[model_path]


service = PredictionService(DataPreprocessor(
    "GoogleNews-vectors-negative300.bin"))


@app.route('/train', methods=['POST'])
def train_model():
    data = request.files['data']
    model_path = request.form['model_path']
    trainer = ModelTrainer(service.preprocessor)
    trainer.train(data, model_path)
    return jsonify({"status": "success", "model_path": model_path})


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    model_path = data['model_path']
    text = data['text']

    model, label_encoder = service.load_model(model_path)
    vector = service.preprocessor.text_to_vectors(text)
    prediction = model.predict(np.array([vector]))

    inv_encoder = {v: k for k, v in label_encoder.items()}
    predicted_class = inv_encoder[np.argmax(prediction)]

    return jsonify({
        "text": text,
        "prediction": predicted_class,
        "confidence": float(np.max(prediction))
    })


if __name__ == '__main__':
    app.run(port=5000, debug=True)
