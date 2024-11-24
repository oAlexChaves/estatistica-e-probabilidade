import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # Importando Matplotlib para os gráficos
import seaborn as sns

# Carregar os dados
df = pd.read_csv('unicorn_companies.csv')

# Função auxiliar para converter valores
def converter_valor(valor_str):
    if pd.isna(valor_str):
        return 0
    valor_str = valor_str.strip().replace('$', '')
    
    # Verificando e tratando diferentes sufixos de valor
    if valor_str.endswith('B'):
        return float(valor_str.replace('B', '')) * 1e9
    elif valor_str.endswith('M'):
        return float(valor_str.replace('M', '')) * 1e6
    elif valor_str.endswith('K'):
        return float(valor_str.replace('K', '')) * 1e3
    else:
        return float(valor_str)  # Se não tiver sufixo, tenta converter diretamente para float

df['post_money_value'] = df['post_money_value'].apply(converter_valor)
df['total_eq_funding'] = df['total_eq_funding'].apply(converter_valor)

# Funções existentes
def probabilidade_simples():
    continentes_counts = df['region'].value_counts()
    continente_mais_repetido = continentes_counts.idxmax()
    quantidade = continentes_counts.max()
    total_unicornios = continentes_counts.sum()
    porcentagem_mais_repetido = (quantidade / total_unicornios) * 100
    print(f"O continente com mais unicórnios emergentes é {continente_mais_repetido}, com {quantidade} unicórnios ({porcentagem_mais_repetido:.2f}%).")
    print("\nRanking de unicórnios por continente:")
    for continente, count in continentes_counts.items():
        print(f"{continente}: {count} unicórnios")

def probabilidade_condicional():
    empresas_valiosas_1b = df[df['post_money_value'] > 1e9]
    empresas_valiosas_100b = df[df['post_money_value'] > 100e9]

    if not empresas_valiosas_1b.empty:
        continente_mais_frequente_1b = empresas_valiosas_1b['region'].value_counts(normalize=True).idxmax()
        probabilidade_continente_1b = empresas_valiosas_1b['region'].value_counts(normalize=True).max()
        print(f"Continente com maior % de empresas com valuation > 1 bilhão: {continente_mais_frequente_1b} ({probabilidade_continente_1b:.2%})")
    else:
        print("Não há empresas com valuation superior a 1 bilhão.")

    if not empresas_valiosas_100b.empty:
        continente_mais_frequente_100b = empresas_valiosas_100b['region'].value_counts(normalize=True).idxmax()
        probabilidade_continente_100b = empresas_valiosas_100b['region'].value_counts(normalize=True).max()
        print(f"Continente com maior % de empresas com valuation > 100 bilhões: {continente_mais_frequente_100b} ({probabilidade_continente_100b:.2%})")
    else:
        print("Não há empresas com valuation superior a 100 bilhões.")

def valores_e_frequencias_lead_investors():
    lead_investors_counts = df['lead_investors'].value_counts()
    print("Valores e suas contagens na coluna 'lead_investors':")
    print(lead_investors_counts)

def probabilidade_investidores_dada_regiao_bayes(regiao, investidor):
    total_unicornios = df.shape[0]
    unicornios_com_investidor = df[df['lead_investors'] == investidor].shape[0]
    prob_investidor = unicornios_com_investidor / total_unicornios
    unicornios_na_regiao = df[df['region'] == regiao].shape[0]
    prob_regiao = unicornios_na_regiao / total_unicornios
    df_investidor = df[df['lead_investors'] == investidor]
    unicornios_regiao_investidor = df_investidor[df_investidor['region'] == regiao].shape[0]
    prob_regiao_dado_investidor = unicornios_regiao_investidor / unicornios_com_investidor if unicornios_com_investidor > 0 else 0

    if prob_regiao > 0:
        prob_investidor_dado_regiao = (prob_regiao_dado_investidor * prob_investidor) / prob_regiao
        print(f"A probabilidade de um unicórnio na região '{regiao}' ter o investidor '{investidor}' é {prob_investidor_dado_regiao:.2%}.")
    else:
        print(f"Não há unicórnios na região '{regiao}'.")
        
def analise_variaveis_aleatorias(n_amostras=1000):
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
    valuations = df['post_money_value'].dropna().values
    random_samples = np.random.choice(valuations, size=n_amostras, replace=True)
    media = np.mean(random_samples)
    mediana = np.median(random_samples)
    desvio_padrao = np.std(random_samples)
    print(f"Análise de Valuations (amostragem de {n_amostras} unicórnios):")
    print(f"  - Média: ${media:,.2f}")
    print(f"  - Mediana: ${mediana:,.2f}")
    print(f"  - Desvio Padrão: ${desvio_padrao:,.2f}")
    prob_10b = np.mean(random_samples > 10e9)
    print(f"Probabilidade de um unicórnio aleatório ter valuation > $10B: {prob_10b:.2%}")

# Nova função: Comparação entre regiões com box plot
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

# Função para plotar histograma geral dos valuations
def histograma_geral_valuation():
    df_filtrado = df[df['post_money_value'] > 0]['post_money_value']
    
    plt.figure(figsize=(10, 6))
    plt.hist(df_filtrado, bins=30, color='skyblue', edgecolor='black', log=True)
    plt.title('Distribuição Geral de Valuation das Empresas Unicórnios')
    plt.xlabel('Valuation (USD)')
    plt.ylabel('Frequência (Escala Logarítmica)')
    plt.xscale('log')
    plt.grid(True)
    plt.show()

def histograma_unicornios_por_regiao():
    # Contar quantos unicórnios existem em cada região
    contagem_regioes = df['region'].value_counts()

    # Plotar o gráfico de barras
    plt.figure(figsize=(10, 6))
    plt.bar(contagem_regioes.index, contagem_regioes.values, color='skyblue', edgecolor='black')
    plt.title('Número de Unicórnios por Região')
    plt.xlabel('Região')
    plt.ylabel('Número de Unicórnios')
    plt.xticks(rotation=45)  # Rotaciona os nomes das regiões para melhor visualização
    plt.grid(axis='y')
    
    plt.show()

def scatter_funding_vs_valuation():
    # Função para converter valores em formato string para float
    def converter_valor(valor_str):
        if pd.isna(valor_str):
            return 0
        valor_str = valor_str.strip().replace('$', '').replace(',', '')
        if valor_str.endswith('B'):
            return float(valor_str[:-1]) * 1e9
        elif valor_str.endswith('M'):
            return float(valor_str[:-1]) * 1e6
        return float(valor_str)
    
    # Aplicar a conversão para colunas relevantes
    df['post_money_value'] = df['post_money_value'].apply(converter_valor)
    df['total_eq_funding'] = df['total_eq_funding'].apply(converter_valor)
    
    # Filtrar dados válidos
    df_plot = df[(df['post_money_value'] > 0) & (df['total_eq_funding'] > 0)]
    
    # Criar o scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df_plot['total_eq_funding'], df_plot['post_money_value'],
                alpha=0.7, c='green', edgecolors='black')
    
    # Configurações do gráfico
    plt.title('Total de Investimentos vs. Valuation')
    plt.xlabel('Total de Investimentos (USD)')
    plt.ylabel('Valuation Pós-Investimento (USD)')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.show()


def distribuicao_valuation_por_regiao_investidor():
    # Criando variáveis dummy para região e investidores principais
    df_dummies = pd.get_dummies(df[['region', 'lead_investors']], drop_first=True)
    
    # Combinando os valores do valuation com as variáveis categóricas
    df_combined = pd.concat([df[['post_money_value']], df_dummies], axis=1)
    
    # Calculando a média do valuation para cada combinação de região e investidor
    df_grouped = df_combined.groupby(df_dummies.columns.tolist())['post_money_value'].mean().unstack()
    
    # Criando o heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_grouped, annot=True, cmap='coolwarm', fmt='.2f', cbar=True, annot_kws={"size": 10})
    plt.title('Distribuição de Valuation por Região e Investidor Principal', fontsize=14)
    plt.xlabel('Investidor Principal')
    plt.ylabel('Região')
    plt.show()

# Menu interativo com a nova opção
while True:
    print("\nMenu de Funções:")
    print("1. Probabilidade Simples")
    print("2. Probabilidade Condicional")
    print("3. Valores e Frequências (Lead Investors)")
    print("4. Probabilidade Bayesiana (Investidor | Região)")
    print("5. Análise de Variáveis Aleatórias")
    print("6. Comparação entre Regiões (Box Plot)")
    print("7. Histograma Geral de Valuation")
    print("8. Histograma de Unicórnios por Região")
    print("9. Scatter Plot (Ano de Fundação vs Valuation)")
    print("10. Distribuição de Valuation por Região e Investidor")
    print("11. Sair")
    
    escolha = input("Escolha uma opção (1-12): ")
    
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
        comparacao_boxplot()
    elif escolha == '7':
        histograma_geral_valuation()
    elif escolha == '8':
        histograma_unicornios_por_regiao()
    elif escolha == '9':
        scatter_funding_vs_valuation()
    elif escolha == '10':
        distribuicao_valuation_por_regiao_investidor()  # Chama a nova função para o heatmap
    elif escolha == '11':
        print("Encerrando o programa. Até mais!")
        break
    else:
        print("Opção inválida. Tente novamente.")