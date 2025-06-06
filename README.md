# Inventory Algorithm Flask API

A Flask-based backend for inventory forecasting, featuring endpoints for SARIMAX time series models, XGBoost (planned), and DistilBERT-based NLP sentiment analysis.  
This API is designed for integration with frontend dashboards or automated forecasting pipelines.

---

## Project Structure

```
.
├── app.py                   # Main Flask app and routing
├── services/
│   ├── sarimax_service.py   # SARIMAX model training & prediction logic
│   ├── xgboost_service.py   # XGBoost service (placeholder)
│   └── nlp_service.py       # DistilBERT NLP service (TensorFlow, HuggingFace)
├── models/                  # Serialized (trained) models go here
│   ├── sarimax_model.pkl
│   └── distilbert_model/
├── requirements.txt
└── README.md
```

---

## API Endpoints

- `POST /sarimax/train`  
  Train a SARIMAX model.  
  **Body:** JSON with `y` (time series), `order`, `seasonal_order`, and optionally `exog` for exogenous variables.

- `POST /sarimax/predict`  
  Make predictions using a trained SARIMAX model.  
  **Body:** JSON with `steps` (forecast horizon), optionally `exog_future`.

- `POST /xgboost/train` & `/xgboost/predict`  
  (Placeholder for future XGBoost support.)

- `POST /nlp/predict`  
  Get sentiment predictions using a pre-trained DistilBERT model.  
  **Body:** JSON with `texts` (list of strings).

---

## Quickstart

1. **Install requirements**
    ```bash
    pip install -r requirements.txt
    ```

2. **Start the API server**
    ```bash
    python app.py
    ```

3. **Example: Train SARIMAX**
    ```bash
    curl -X POST http://localhost:5000/sarimax/train \
      -H "Content-Type: application/json" \
      -d '{"y": [10, 12, 15, ...], "order": [1,1,1], "seasonal_order": [0,1,1,7]}'
    ```

4. **Example: Predict Sentiment with NLP**
    ```bash
    curl -X POST http://localhost:5000/nlp/predict \
      -H "Content-Type: application/json" \
      -d '{"texts": ["The product is great!", "This is a bad experience."]}'
    ```

---

## Model Management

- Place your trained model files in the `models/` directory:
  - SARIMAX: `models/sarimax_model.pkl`
  - DistilBERT: `models/distilbert_model/` (directory with tokenizer and model weights)
- The API expects these files to be present before serving predictions.

---

## Development Notes

- XGBoost support is planned for future releases.
- The NLP service requires a pre-trained DistilBERT model (TensorFlow/HuggingFace).
- To extend with new exogenous features (e.g., sentiment scores from NLP), update both the SARIMAX training and prediction endpoints and logic.

### Future Direction: Integrating NLP Features with SARIMAX

You can further enhance your forecasts by:
- Aggregating daily sentiment scores from `/nlp/predict`.
- Passing these as exogenous variables (`exog`) to SARIMAX endpoints.

---

## License

Specify your license here (MIT, Apache-2.0, etc.).

---

## Acknowledgements

- [Flask](https://flask.palletsprojects.com/)
- [Statsmodels](https://www.statsmodels.org/)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [TensorFlow](https://www.tensorflow.org/)