# 🏡 Ames Housing Price Prediction: Pipeline de Regressão & Interpretabilidade

Projeto de Machine Learning ponta a ponta voltado para a precificação de imóveis residenciais. O projeto abrange desde o tratamento avançado de valores ausentes e detecção de outliers até a comparação de modelos preditivos (Deep Learning, SVM e Random Forest), validação por interpretabilidade com SHAP e deploy de uma arquitetura simplificada voltada para produção.

---
🛠️ Tecnologias Utilizadas
Linguagem: Python

Manipulação & Análise: Pandas, NumPy, KaggleHub

Visualização: Seaborn, Matplotlib

Machine Learning & Pré-processamento: Scikit-Learn (RobustScaler, KNNImputer, OneHotEncoder, RandomForestRegressor, SVR)

Deep Learning: TensorFlow / Keras (Sequential, Dense, BatchNormalization, Dropout, EarlyStopping)

Model Explainability: SHAP, Permutation Importance

## 📌 Destaques do Projeto

- **Tratamento Robusto de Dados:** Imputação por vizinhos mais próximos (`KNNImputer`) combinada com `RobustScaler` para variáveis com alto coeficiente de variação, além de contenção de outliers (*IQR Capping*).
- **Benchmark de Modelos:** Comparação empírica de 3 arquiteturas distintas: **Rede Neural Multicamadas (Keras/TensorFlow)**, **Support Vector Regressor (SVR Polinomial)** e **Random Forest Regressor**.
- **Explicabilidade de Caixa-Preta (XAI):** Avaliação profunda do peso das variáveis utilizando **Gini Importance**, **Permutation Importance** e **SHAP (SHapley Additive exPlanations)**.
- **Engenharia para Produção:** Redução de dimensionalidade de **266 features codificadas para apenas as Top 10 variáveis críticas**, mantendo a acurácia preditiva ($R^2$ retido em ~89%) e viabilizando inferência leve via payload JSON.

---

## 🔬 Benchmark e Comparação de Modelos

Os modelos foram treinados e avaliados no conjunto de teste (20% dos dados) com desescalonamento para mensuração em escala monetária real ($):

| Modelo | MAE ($) | RMSE ($) | $R^2$ (%) |
| :--- | :---: | :---: | :---: |
| **Random Forest Regressor** | **$15.343,84** | **$ 20.957,78** | **91.02%** |
| Support Vector Regressor (Poly) | $16.209,58 | $ 20.957,78 | 88.54%% |
| Rede Neural MLP (Keras) | $14.895,44 | 21.097,42 | 90.90%% |


---

## 🔍 Interpretabilidade e Explainable AI (XAI)

Através do **SHAP Beeswarm Plot** e **Permutation Importance**, foi possível isolar os reais direcionadores de valor dos imóveis:
1. **OverallQual (Qualidade Geral dos Materiais):** O fator de maior peso isolated (~60% de importância relativa). Imóveis com score alto elevam o preço drasticamente.
2. **GrLivArea (Área Útil Construída):** Forte correlação positiva linear com o preço final de venda.
3. **TotalBsmtSF & GarageCars:** A presença de subsolo acabado e capacidade da garagem são os diferenciais estruturais secundários de maior impacto.

---

## 🚀 De 266 para 10 Features: Modelo Simplificado para Produção

Para viabilizar a implantação em APIs REST ou aplicativos mobile sem a sobrecarga de pré-processar 266 colunas esparsas:
* Um modelo reduzido foi retreinado utilizando **apenas as 10 principais variáveis**.
* **Resultado:** O modelo simplificado manteve um **$R^2$ praticamente estável**, provando que 96% do ruído dimensional podia ser eliminado sem perda de poder preditivo.

### Exemplo de Inferência:
```python
exemplo_imovel = {
    'OverallQual': 7.0,     # Qualidade geral (1 a 10)
    'GrLivArea': 1710.0,    # Área útil (sqft)
    'TotalBsmtSF': 856.0,   # Área de subsolo
    'BsmtFinSF1': 706.0,    # Subsolo acabado
    'GarageCars': 2.0,      # Vagas de garagem
    'GarageArea': 548.0,    # Área da garagem
    '1stFlrSF': 856.0,      # Área 1º andar
    'LotArea': 8450.0,      # Área total do terreno
    'YearBuilt': 2003.0,    # Ano de construção
    'YearRemodAdd': 2003.0  # Ano de reforma
}

# Retorno instantâneo do valor previsto
preco_estimado = prever_imovel_simplificado(exemplo_imovel, ...)
print(f"Preço Previsto: ${preco_estimado:,.2f}")
