"""
Testes automatizados de desempenho do modelo de classificação
de dependência digital em smartphones.

Objetivo: garantir que o modelo em produção atenda aos requisitos mínimos
de desempenho antes de ser implantado. Caso o modelo seja substituído,
estes testes impedem a implantação de um modelo inferior.

Execute com: pytest test_modelo.py -v
"""
import warnings
import pytest
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

warnings.filterwarnings('ignore')

# ── Configurações globais ────────────────────────────────────────────────────
URL_DATASET  = './dataset/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv'
CAMINHO_MODELO = './machine learning/modelo_SVM.pkl'
TEST_SIZE    = 0.20
SEED         = 7

# ── Thresholds mínimos de desempenho ────────────────────────────────────────
# Valores definidos com base nos resultados obtidos durante o desenvolvimento.
# Um novo modelo só será aceito se superar estes limites em todas as métricas.
THRESHOLD_ACCURACY = 0.90   # acurácia mínima de 90%
THRESHOLD_F1       = 0.93   # F1-Score mínimo de 93%
THRESHOLD_ROC_AUC  = 0.97   # ROC-AUC mínimo de 97%


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def modelo():
    """Carrega o modelo salvo em disco."""
    return joblib.load(CAMINHO_MODELO)


@pytest.fixture(scope='module')
def dados_teste():
    """
    Carrega o dataset, aplica o mesmo pré-processamento do notebook
    e retorna X_test e y_test para avaliação.
    """
    # Carga
    dataset = pd.read_csv(URL_DATASET)
    df = dataset.copy()

    # Remoção de colunas irrelevantes
    df.drop(columns=['transaction_id', 'user_id', 'addiction_level'], inplace=True)

    # Codificação das variáveis categóricas
    encoder = LabelEncoder()
    for col in ['gender', 'stress_level', 'academic_work_impact']:
        df[col] = encoder.fit_transform(df[col])

    # Separação features / target
    X = df.drop(columns=['addicted_label']).values
    y = df['addicted_label'].values

    # Holdout com mesma semente do notebook
    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        shuffle=True,
        random_state=SEED,
        stratify=y
    )

    return X_test, y_test


@pytest.fixture(scope='module')
def predicoes(modelo, dados_teste):
    """Gera predições e probabilidades para o conjunto de teste."""
    X_test, y_test = dados_teste
    predictions   = modelo.predict(X_test)
    probabilities = modelo.predict_proba(X_test)[:, 1]
    return y_test, predictions, probabilities


# ── Testes ────────────────────────────────────────────────────────────────────

def test_modelo_carrega(modelo):
    """Verifica se o arquivo .pkl é carregado sem erros."""
    assert modelo is not None, 'O modelo não foi carregado corretamente.'


def test_modelo_tem_predict(modelo):
    """Verifica se o modelo possui os métodos necessários para predição."""
    assert hasattr(modelo, 'predict'), \
        'O modelo não possui o método predict.'
    assert hasattr(modelo, 'predict_proba'), \
        'O modelo não possui o método predict_proba.'


def test_acuracia_minima(predicoes):
    """
    Acurácia deve ser >= 90%.
    Garante que o modelo acerta ao menos 9 em cada 10 classificações.
    """
    y_test, predictions, _ = predicoes
    acuracia = accuracy_score(y_test, predictions)
    print(f'\n  Acurácia obtida: {acuracia:.4f} | Threshold: {THRESHOLD_ACCURACY}')
    assert acuracia >= THRESHOLD_ACCURACY, \
        f'Acurácia {acuracia:.4f} abaixo do mínimo exigido ({THRESHOLD_ACCURACY}).'


def test_f1_score_minimo(predicoes):
    """
    F1-Score deve ser >= 93%.
    Garante equilíbrio entre precisão e recall na detecção de uso preocupante.
    """
    y_test, predictions, _ = predicoes
    f1 = f1_score(y_test, predictions)
    print(f'\n  F1-Score obtido: {f1:.4f} | Threshold: {THRESHOLD_F1}')
    assert f1 >= THRESHOLD_F1, \
        f'F1-Score {f1:.4f} abaixo do mínimo exigido ({THRESHOLD_F1}).'


def test_roc_auc_minimo(predicoes):
    """
    ROC-AUC deve ser >= 97%.
    Garante excelente capacidade de separação entre as classes.
    É a métrica mais importante para este problema de saúde digital.
    """
    y_test, _, probabilities = predicoes
    roc_auc = roc_auc_score(y_test, probabilities)
    print(f'\n  ROC-AUC obtido: {roc_auc:.4f} | Threshold: {THRESHOLD_ROC_AUC}')
    assert roc_auc >= THRESHOLD_ROC_AUC, \
        f'ROC-AUC {roc_auc:.4f} abaixo do mínimo exigido ({THRESHOLD_ROC_AUC}).'


def test_predicao_perfil_preocupante(modelo):
    """
    Teste de sanidade: perfil claramente preocupante deve ser classificado como 1.
    Usuário com 11h de tela, 5h em redes sociais, 5h de sono.
    """
    perfil_preocupante = pd.DataFrame([[
        19,     # age
        1,      # gender (Masculino)
        11.0,   # daily_screen_time_hours
        5.0,    # social_media_hours
        3.0,    # gaming_hours
        1.0,    # work_study_hours
        5.0,    # sleep_hours
        220,    # notifications_per_day
        150,    # app_opens_per_day
        13.0,   # weekend_screen_time
        0,      # stress_level (Alto)
        0,      # academic_work_impact (Não)
    ]])
    predicao = modelo.predict(perfil_preocupante)[0]
    assert predicao == 1, \
        'Perfil claramente preocupante deveria ser classificado como 1.'


def test_predicao_perfil_saudavel(modelo):
    """
    Teste de sanidade: perfil claramente saudável deve ser classificado como 0.
    Usuário com 3h de tela, 8h de sono, foco em trabalho/estudo.
    """
    perfil_saudavel = pd.DataFrame([[
        28,     # age
        0,      # gender (Feminino)
        3.0,    # daily_screen_time_hours
        1.0,    # social_media_hours
        0.5,    # gaming_hours
        6.0,    # work_study_hours
        8.0,    # sleep_hours
        40,     # notifications_per_day
        25,     # app_opens_per_day
        4.0,    # weekend_screen_time
        1,      # stress_level (Baixo)
        1,      # academic_work_impact (Sim)
    ]])
    predicao = modelo.predict(perfil_saudavel)[0]
    assert predicao == 0, \
        'Perfil claramente saudável deveria ser classificado como 0.'


def test_saida_binaria(predicoes):
    """Verifica se todas as predições são binárias (apenas 0 ou 1)."""
    _, predictions, _ = predicoes
    valores_unicos = set(predictions)
    assert valores_unicos.issubset({0, 1}), \
        f'O modelo retornou valores inesperados: {valores_unicos}'


def test_probabilidades_validas(predicoes):
    """Verifica se todas as probabilidades estão entre 0 e 1."""
    _, _, probabilities = predicoes
    assert np.all(probabilities >= 0) and np.all(probabilities <= 1), \
        'O modelo retornou probabilidades fora do intervalo [0, 1].'
