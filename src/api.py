#!/usr/bin/env python3
"""
Flask API - Rent Prediction Service with Swagger
"""

from flask import Flask, request, jsonify
from flasgger import Swagger, swag_from
from datetime import datetime
import traceback

# RentPredictor sınıfını içeri aktar
from predict import RentPredictor, create_sample_data

app = Flask(__name__)
swagger = Swagger(app)  # Swagger başlat

# Global predictor nesnesi
predictor = RentPredictor()


@app.route("/", methods=["GET"])
def home():
    """Ana sayfa - API bilgileri
    ---
    responses:
      200:
        description: API bilgilerini döndürür
    """
    return jsonify({
        "service": "Rent Prediction API",
        "version": "1.0",
        "description": "Hamburg 50 km cevresi kira tahmin servisi",
        "available_endpoints": {
            "GET /": "API bilgileri",
            "POST /predict": "Kira tahmini yap",
            "GET /sample": "Örnek veri formatı"
        },
        "timestamp": datetime.now().isoformat()
    })


@app.route("/predict", methods=["POST"])
@swag_from({
    'tags': ['Prediction'],
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'city': {'type': 'string'},
                    'district': {'type': 'string'},
                    'object_age': {'type': 'number'},
                    'flat_area': {'type': 'number'},
                    'room_count': {'type': 'integer'},
                    'distance_to_centre': {'type': 'number'}
                },
                'required': [
                    'city', 'district', 'object_age',
                    'flat_area', 'room_count', 'distance_to_centre'
                ]
            }
        }
    ],
    'responses': {
        200: {
            'description': 'Tahmin sonucu',
            'examples': {
                'application/json': {
                    "status": "success",
                    "input": {},
                    "prediction": 1200,
                    "prediction_text": "Tahmini kira",
                    "probability": 1.0,
                    "timestamp": "2025-10-19T12:34:56"
                }
            }
        },
        400: {'description': 'Geçersiz JSON'},
        500: {'description': 'Sunucu hatası'}
    }
})
def predict():
    """Kira tahmini yap"""
    try:
        if not request.is_json:
            return jsonify({"error": "JSON formatında veri gönderin"}), 400

        data = request.get_json()
        result = predictor.predict(data)

        return jsonify({
            "status": "success",
            "input": data,
            "prediction": result["prediction"],
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        print("❌ Tahmin hatası:", e)
        print(traceback.format_exc())
        return jsonify({"error": "İşlem sırasında hata oluştu"}), 500


@app.route("/sample", methods=["GET"])
def sample():
    """Örnek istek formatı
    ---
    responses:
      200:
        description: Örnek veri döndürür
    """
    return jsonify({
        "description": "Örnek istek formatı",
        "sample_request": {
            "city": "Berlin",
            "district": "Mitte",
            "cold_price": 1200,
            "object_age": 10,
            "flat_area": 60,
            "room_count": 2,
            "distance_to_centre": 3.5
        },
        "required_fields": [
            "city", "district", "cold_price", "object_age",
            "flat_area", "room_count", "distance_to_centre"
        ]
    })


def main():
    """Flask uygulamasını başlat"""
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,  # Production'da False olmalı
        threaded=True
    )


if __name__ == "__main__":
    print("✅ Rent Prediction API başladı: http://localhost:5000")
    main()
