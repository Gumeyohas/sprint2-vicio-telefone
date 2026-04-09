# PhoneCheck — Classificação de Dependência Digital

MVP da Sprint 2 — Engenharia de Software | PUC-Rio

## Sobre o Projeto

Aplicação full stack que utiliza um modelo de Machine Learning (SVM)
para classificar o padrão de uso digital de um usuário como
**equilibrado** ou **preocupante**, com base em dados de uso do smartphone.

## Estrutura do Repositório

- `notebook/` — Notebook Google Colab com todo o processo de criação do modelo
- `backend/` — API REST em Flask com o modelo embarcado
- `frontend/` — Interface web para entrada de dados e exibição do resultado

## Como Executar

### Back-end
```bash
cd backend
pip install -r requirements.txt
python app.py
```

### Front-end
Abra o arquivo `frontend/index.html` diretamente no navegador.

### Testes
```bash
cd backend
.venv/bin/python -m pytest test_modelo.py -v
```

## Modelo de Machine Learning

- **Algoritmo**: SVM (kernel=rbf, C=1, gamma=scale)
- **Pré-processamento**: StandardScaler
- **Métricas no conjunto de teste**:
  - Acurácia: 92.33%
  - F1-Score: 0.947
  - ROC-AUC: 0.974
- **Dataset**: 7.500 registros de uso de smartphone