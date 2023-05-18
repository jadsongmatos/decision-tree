# %% [markdown]
# # **Classificação com Árvore de Decisão**
# 

# %% [markdown]
# https://www.kaggle.com/datasets/rever3nd/weather-data?resource=download
# 

# %%
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder

# %%
# Carregue os dados meteorológicos do arquivo CSV
df = pd.read_csv("weather.csv")


# %%
# Elimine as linhas com valores ausentes e redefina o índice
df = df.dropna().reset_index(drop=True)


# %%
# Use LabelEncoder para codificar variáveis categóricas para valores numéricos
le_sex = LabelEncoder()
inputs = df.drop(['RainTomorrow', 'Unnamed: 0'], axis='columns')
inputs['Location'] = le_sex.fit_transform(inputs['Location'])
inputs['WindGustDir'] = le_sex.fit_transform(inputs['WindGustDir'])
inputs['RainToday'] = le_sex.fit_transform(inputs['RainToday'])
inputs['WindDir9am'] = le_sex.fit_transform(inputs['WindDir9am'])
inputs['WindDir3pm'] = le_sex.fit_transform(inputs['WindDir3pm'])


# Convertendo a coluna 'data' para o tipo datetime
inputs['Date'] = pd.to_datetime(inputs['Date'])
# Convertendo a coluna 'data' para número usando o método astype
inputs['Date'] = inputs['Date'].astype(int)


# %%
# Prepare a variável de destino codificando a coluna 'RainTomorrow'
target = le_sex.fit_transform(df['RainTomorrow'])


# %%
# Crie uma instância do DecisionTreeClassifier
model = tree.DecisionTreeClassifier()

# Treine o modelo de árvore de decisão
model.fit(inputs, target)

# %%
# Calculate the accuracy score of the model on the training data
print("score:",model.score(inputs, target))