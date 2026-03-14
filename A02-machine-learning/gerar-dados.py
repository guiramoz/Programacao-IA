#importanto as libs necessárias
import pandas as pd
import numpy as np

#criando numeros aleatórios para simular dados reais
#definindo uma semente para fins de simulação
np.random.seed(42)

#gerando 500 registros
n_registros = 500

#estruturando os dados do arquivo .csv
data = {
    #1 a 48 meses, randint: pega numeros aleatorios
    'tempo_contrato': np.random.randint(1, 48, n_registros), 
    #assinatura com valores que variam de 50 a 150 dinheiros, uniform: distribuição uniforme
    'valor_mensal': np.random.uniform(50.0, 150.0,n_registros).round(2), 
    #cada usuário tem uma média de 1.5 reclamações, poison: sorteio de eventos,exemplo de uma fila de banco
    'reclamacoes': np.random.poisson(1.5,n_registros)  
}

#corventendo a estrutura de dicionário em um conjunto de dados
df = pd.DataFrame(data)

#criar a simulação da lógica de churn
#o cliente tem mais chance de sair se tiver muitsa reclamações OU
#se o contrato for curto
df['cancelou']=((df['reclamacoes']>2)|(df['tempo_contrato']<6)).astype(int)

#salvar o dataset em .csv
df.to_csv('churn-data.csv', index = False)
print("Arquivo 'churn-data.csv' gerado com sucesso!")