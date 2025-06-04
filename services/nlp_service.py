import os
import numpy as np
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf

class NLPService:
    def __init__(self, model_path='models/distilbert_model'):
        self.model_path = model_path
        self.model, self.tokenizer = self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            model = TFDistilBertForSequenceClassification.from_pretrained(self.model_path)
            tokenizer = DistilBertTokenizer.from_pretrained(self.model_path)
        else:
            model = None
            tokenizer = None
        return model, tokenizer

    def predict(self, data):
        if self.model is None or self.tokenizer is None:
            return {'error': 'DistilBERT model not loaded.'}
        texts = data.get('texts', [])
        inputs = self.tokenizer(texts, return_tensors='tf', padding=True, truncation=True)
        outputs = self.model(inputs)
        predictions = tf.math.argmax(outputs.logits, axis=1).numpy().tolist()
        return {'predictions': predictions}