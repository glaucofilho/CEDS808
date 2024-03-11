import warnings

import pandas as pd
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# Suprimindo warnings de convergência
warnings.filterwarnings("ignore", category=ConvergenceWarning)
# dados originais
# https://www.kaggle.com/datasets/renangomes/dengue-temperatura-e-chuvas-em-campinassp/data
# Chuvas faltando para
# completando dados com
# https://www.cpa.unicamp.br/graficos
# 1999-07-01 2
# 1999-08-01 0
# 2002-06-01 0
# 2004-08-01 0
# 2007-08-01 0
# 2008-07-01 191
# 2010-08-01 0
# 2012-08-01 0

# adicionando mais uma feature, populacao
# baseado nos sensos
# 1991-01-01  847595
# 2000-01-01  969396
# 2010-01-01  1080999
# 2022-01-01  1139047


populacao = {
    pd.Timestamp("1991-01-01"): 847595,
    pd.Timestamp("2000-01-01"): 969396,
    pd.Timestamp("2010-01-01"): 1080999,
    pd.Timestamp("2022-01-01"): 1139047,
}


# Carregando os dados tipando cada coluna
df = pd.read_csv(
    "dengue-dataset.csv",
    dtype={
        "data": str,
        "casos-confirmados": float,
        "chuva": float,
        "temperatura-media": float,
        "temperatura-mininima": float,
        "temperatura-maxima": float,
    },
)

# Interpolando linearmente a populacao de campinas
df["data"] = pd.to_datetime(df["data"])
df["populacao"] = df["data"].map(populacao)
df["populacao"] = df["populacao"].interpolate(method="linear")

# com a adicao da populacao tive uma melhora de 14% de acuracia para 55%

# removendo dados falsos de 1991 e 2022
df.drop(index=[0, 205], inplace=True)


# adicionando mais uma feature, investimento em saude corrijido pela inflacao
# fonte LOA (lei orcamentaria anual) e noticias de jornal
# sem correcao inflacao IPCA
# 2015: 1115767005.00, correcao x1,67
# 2014: 1027369524.00, correcao x1,78
# 2013: 1000262264.00, correcao x1,88
# 2012: 846603134.00, correcao x1,99
# 2011:  807363166.00, correcao x2,12
# 2010: 717517988.00, correcao x2,24
# 2009: 654000000.00, correcao x2,34
# 2008: 518200000.00, correcao x2,48
# 2007: 445000000.00, correcao x2,59
# 2006: 400000000.00, correcao x2,67
# 2005: ?, correcao x2,82
# 2004: ?, correcao x3,04
# 2003: ?, correcao x3,32
# 2002: 218588697.00, correcao x3,74
# 2001: , correcao x4,02
# 2000: 188000000.00, correcao x4,26
# 1999: , correcao x4,64
# 1998: , correcao x4,72
# Dados de investimento em saúde e suas correções
# 2005, 2004, 2003, 2001, 1999 e 1998 foram feitos com interpolacao

investimento_saude = {
    2015: 1115767005.00 * 1.67,
    2014: 1027369524.00 * 1.78,
    2013: 1000262264.00 * 1.88,
    2012: 846603134.00 * 1.99,
    2011: 807363166.00 * 2.12,
    2010: 717517988.00 * 2.24,
    2009: 654000000.00 * 2.34,
    2008: 518200000.00 * 2.48,
    2007: 445000000.00 * 2.59,
    2006: 400000000.00 * 2.67,
    2002: 218588697.00 * 3.74,
    2000: 188000000.00 * 4.26,
    2005: 1008332363.695,
    2004: 939141044.39,
    2003: 880141045.085,
    2001: 809200863.39,
    1999: 792559136.61,
    1998: 784238273.22,
}

# Transformando a coluna data para mês e ano inteiro
df["mes"] = df["data"].dt.month
df["ano"] = df["data"].dt.year
df["mes"] = df["mes"].astype(int)
df["ano"] = df["ano"].astype(int)
del df["data"]

# Adicionando o investimento em saude anual
df["investimento_saude"] = df["ano"].map(investimento_saude)

# Remoção de Outliers
df = df.drop(df[df["casos-confirmados"] > 5580].index)

# Lidando com valores ausentes imputando a média por mês
df["chuva"] = df.groupby("mes")["chuva"].transform(
    lambda x: x.fillna(x.mean())
)

# Definindo features e target
X = df.copy()
del X["casos-confirmados"]
y = df["casos-confirmados"]

# Padronizando os dados
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Modelos a serem testados
models = {
    "KNN": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "SVM": SVR(),
    "Multi-layer Perceptron": MLPRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "XGBoost": xgb.XGBRegressor(),
}

# Parâmetros a serem testados para cada modelo
params = {
    "KNN": {"n_neighbors": [3, 5, 7, 9, 11, 13, 15]},
    "Decision Tree": {"max_depth": [None, 5, 10, 15, 20, 25, 30]},
    "SVM": {"C": [0.1, 1, 10, 100], "epsilon": [0.1, 0.2, 0.5, 1]},
    "Multi-layer Perceptron": {
        "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 100)]
    },
    "Random Forest": {
        "n_estimators": [50, 100, 150, 200],
        "max_depth": [None, 5, 10, 15, 20],
    },
    "Gradient Boosting": {
        "n_estimators": [50, 100, 150, 200],
        "learning_rate": [0.01, 0.1, 0.5, 1],
    },
    "XGBoost": {
        "n_estimators": [50, 100, 150, 200],
        "learning_rate": [0.01, 0.1, 0.5, 1],
    },
}

# Realizando Grid Search com validação cruzada
for name, model in models.items():
    print(f"Training {name}...")
    grid_search = GridSearchCV(model, params[name], cv=5, scoring="r2")
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best {name} R2 Score: {grid_search.best_score_:.3f}")
    predictions = best_model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"{name} Mean Squared Error:", mse)
    print(f"{name} R2 Score:", r2)
    print()
