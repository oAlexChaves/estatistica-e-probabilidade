import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Carregar os dados
df = pd.read_csv('unicorn_companies.csv')

# Função auxiliar para converter valores
def converter_valor(valor_str):
    if pd.isna(valor_str):
        return 0
    valor_str = valor_str.strip().replace('$', '')
    if valor_str.endswith('B'):
        return float(valor_str.replace('B', '')) * 1e9
    elif valor_str.endswith('M'):
        return float(valor_str.replace('M', '')) * 1e6
    else:
        return float(valor_str)

df['post_money_value'] = df['post_money_value'].apply(converter_valor)

# Nova função: Comparação entre regiões ou países com box plot usando Matplotlib
def comparacao_boxplot():
    plt.figure(figsize=(12, 6))
    
    # Filtrando dados sem valuation zero
    df_filtrado = df[df['post_money_value'] > 0]
    
    # Criando o boxplot por região
    df_filtrado.boxplot(column='post_money_value', by='region', patch_artist=True)
    
    plt.yscale('log')  # Escala logarítmica para melhor visualização
    plt.title('Distribuição de Valuation por Região')
    plt.suptitle('')  # Remover título padrão automático do Matplotlib
    plt.xlabel('Região')
    plt.ylabel('Valuation (Escala Logarítmica)')
    plt.xticks(rotation=45)  # Rotaciona os nomes das regiões
    
    plt.show()

# Menu interativo com a nova opção
while True:
    print("\nMenu de Funções:")
    print("1. Probabilidade Simples")
    print("2. Probabilidade Condicional")
    print("3. Valores e Frequências (Lead Investors)")
    print("4. Probabilidade Bayesiana (Investidor | Região)")
    print("5. Análise de Variáveis Aleatórias")
    print("6. Comparação entre Regiões (Box Plot)")  # Nova opção no menu
    print("7. Sair")
    
    escolha = input("Escolha uma opção (1-7): ")
    
    if escolha == '1':
        probabilidade_simples()
    elif escolha == '2':
        probabilidade_condicional()
    elif escolha == '3':
        valores_e_frequencias_lead_investors()
    elif escolha == '4':
        regiao = input("Informe a região: ")
        investidor = input("Informe o investidor: ")
        probabilidade_investidores_dada_regiao_bayes(regiao, investidor)
    elif escolha == '5':
        n_amostras = int(input("Informe o número de amostras: "))
        analise_variaveis_aleatorias(n_amostras)
    elif escolha == '6':
        comparacao_boxplot()  # Chama a nova função
    elif escolha == '7':
        print("Encerrando o programa. Até mais!")
        break
    else:
        print("Opção inválida. Tente novamente.")
