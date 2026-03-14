#ETAPA 1:importando todos os módulos necessários

import pandas as pd #ferramenta para criar e alterar dados em tabelas
import numpy as np #ferramenta de análise matemática

from sklearn.preprocessing import StandardScaler #organiza os números, faz potência de 10, para deixar os números grandes pequenos pois a IA tem dificuldade
from sklearn.ensemble import RandomForestClassifier #organiza o pensamento da IA
from sklearn.metrics import classification_report, confusion_matrix #faz um relatório de efiência
from sklearn.model_selection import train_test_split #permite que o modelo organize 80% para treinar e 20% para testar

import seaborn as sns
import matplotlib.pyplot as pyplot
import joblib #salva o agente 

#ETAPA 2: importação do dataset

try:
    print("Carregando arquivo 'churn-data.csv'...")
    df = pd.read_csv('churn-data.csv')#ler o arquivo e criar uma tabela
    print(f"Sucesso, {len(df)} linhas importadas.")
    
except FileNotFoundError:
    print("O arquivo não pode ser encontrado na pasta.")
    exit()
    
#ETAPA 3: pre processamento de dados (preparar a IA para ser treinada)
#passo 1: separar pergunta (x) da resposta (y)
# (x) -> tudo menos a coluna cancelou, são as "pistas" pro modelo
X=df.drop('cancelou', axis=1)
# (y) -> apenas a coluna 'cancelou', é o que queremos que o modelo preveja
y = df['cancelou']

#passo 2: dividir o treino do teste
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
#test_size=0.2 separa 20% da massa de dados para testar o modelo

#passo 3: normalizando (colocando tudo na mesma escala)
scaler = StandardScaler()

#fit transform do treino: IA calcula a média de desvio padrão do treino
X_train_scaled = scaler.fit_transform(X_train)

#transform no teste: usamos a régua calculada no treino
X_test_scaled = scaler.transform(X_test)

#ETAPA 4: treinar o modelo e realizar a previsão de dados
#criando modelo
modelo_churn = RandomForestClassifier(n_estimators=100, random_state=42)

#treinar/ajustar a IA
modelo_churn.fit(X_train_scaled, y_train)

#prever as respostas
previsoes = modelo_churn.predict(X_test_scaled)

#ETAPA 5: avaliação do modelo
print("Relatório de performance")
print(classification_report(y_test, previsoes))

#ETAPA 6: Deploy -> salvar o trabalho
joblib.dump(modelo_churn,'modelo-churn_v1.pkl')

joblib.dump(scaler,'padronizador_v1.pkl')
print("Arquivos de ML foram exportador com sucesso")