import warnings
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

warnings.filterwarnings('ignore', category=UserWarning);

# ── Inicialização ────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # permite que o front-end acesse o back-end

# ── Carrega o modelo UMA vez ao iniciar o servidor ───────
modelo = joblib.load('./machine learning/modelo_SVM.pkl')

# ── Rota de predição ─────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    dados = request.get_json()

    entrada = pd.DataFrame([[
        dados['age'],
        dados['gender'],
        dados['daily_screen_time_hours'],
        dados['social_media_hours'],
        dados['gaming_hours'],
        dados['work_study_hours'],
        dados['sleep_hours'],
        dados['notifications_per_day'],
        dados['app_opens_per_day'],
        dados['weekend_screen_time'],
        dados['stress_level'],
        dados['academic_work_impact'],
    ]])

    predicao    = modelo.predict(entrada)[0]
    probabilidade = modelo.predict_proba(entrada)[0]

    return jsonify({
        'preocupante'        : int(predicao),
        'prob_equilibrado': round(float(probabilidade[0]) * 100, 2),
        'prob_preocupante'   : round(float(probabilidade[1]) * 100, 2)
    })

# ── Rota de saúde (teste) ────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(port=5000)