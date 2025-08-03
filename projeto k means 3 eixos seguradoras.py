# -*- coding: utf-8 -*-
"""
Análise de Clusters de Seguradoras - Carga de Bibliotecas
"""

# =============================================================================
# 1. BIBLIOTECAS PADRÃO DO PYTHON
# =============================================================================
import os
import warnings
import datetime
from itertools import cycle
import multiprocessing

# =============================================================================
# 2. MANIPULAÇÃO DE DADOS E COMPUTAÇÃO CIENTÍFICA
# =============================================================================
import numpy as np
import pandas as pd
from pandas import read_excel, set_option
from pandas.plotting import scatter_matrix

# =============================================================================
# 3. VISUALIZAÇÃO DE DADOS
# =============================================================================
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
import mplcursors

# =============================================================================
# 4. APRENDIZADO DE MÁQUINA E ESTATÍSTICA
# =============================================================================
# Pré-processamento
from sklearn.preprocessing import StandardScaler, RobustScaler

# Clusterização
from sklearn import cluster
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation
from sklearn.metrics import adjusted_mutual_info_score
from sklearn import covariance, manifold
from sklearn.decomposition import PCA
from sklearn import metrics

# Clusterização hierárquica
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
from scipy.spatial.distance import pdist

# =============================================================================
# 5. CONFIGURAÇÕES GERAIS
# =============================================================================
# Configurações de exibição
# pd.set_option('display.max_columns', 50)
# plt.style.use('seaborn')
# sns.set_style("whitegrid")

# Configurações de desempenho
os.environ["LOKY_MAX_CPU_COUNT"] = str(multiprocessing.cpu_count())

# Controle de warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)



# =============================================================================
# A - Carregando o conjunto de dados:
# =============================================================================

#filepath = r'C:\Users\Felip\Desktop\tcc2\projeto kmeans 3 eixos seguradoras.xlsx'
url = "https://view.officeapps.live.com/op/view.aspx?src=https%3A%2F%2Fraw.githubusercontent.com%2Ffelipefalcao89%2Fkmeans-brazil-insurance%2Frefs%2Fheads%2Fmain%2Fprojeto%2520kmeans%25203%2520eixos%2520seguradoras.xlsx&wdOrigin=BROWSELINK"
#dataset1 = pd.read_excel(filepathgit)

url = "https://github.com/felipefalcao89/kmeans-brazil-insurance/raw/refs/heads/main/projeto%20kmeans%203%20eixos%20seguradoras.xlsx"
dataset1 = pd.read_excel(url, engine='openpyxl')





# Para Análises de ramos de mercado - 
dataset1 = dataset1[dataset1['Ramo'] == 'Vida em Grupo'] #### Selecione aqui o Ramo


print(dataset1.head(5))

# =============================================================================
# A.1 - Removendo as colunas categóricas do Modelo e definindo todos os subsets do modelo, ajustando dados Para modelo
# =============================================================================
#colunas_categoricas = ['Código de Grupo de Ramo', 'Código Ramo Seguro', 'Código Grupo Econômico', 'Grupo Econômico', 'Ramo', 'Grupo De Ramos']
colunas_categoricas = ['Grupo Econômico', 'Ramo', 'Grupo De Ramos']
colunas_calculadas = [
    'Índice de Despesas',
    'Sinistralidade',
    'Índice de Solvência',
    'Índice Combinado',           
    'Market Share 2024',
    'Var Market Share',
    'Participação Relativa de Mercado']

colunas_calculadas_com_classes = [
    'Índice de Despesas',
    'Sinistralidade',
    'Índice de Solvência',
    'Índice Combinado',           
    'Market Share 2024',
    'Var Market Share',
    'Participação Relativa de Mercado',
    'Código de Grupo de Ramo',
    'Código Ramo Seguro',
    'Código Grupo Econômico']

colunas_3_eixos =['Índice de Solvência','Índice Combinado','Participação Relativa de Mercado']

colunas_3_eixos_com_legenda = np.concatenate((colunas_3_eixos, colunas_categoricas))
dataset1['Índice Combinado'] = dataset1['Índice Combinado'] * -1
dataset_completo = dataset1

dataset1 = dataset1.drop(colunas_categoricas, axis=1)

print(dataset1.head(1))

dataset2 = dataset1[colunas_3_eixos]
print(dataset2.head(1))
# =============================================================================
# B - Removendo Outliers:
# =============================================================================


def remove_outliers(df: pd.DataFrame, 
                   threshold: float = 3,
                   verbose: bool = True) -> pd.DataFrame:
    """
    Remove outliers de um DataFrame usando o método Z-score.
    
    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame contendo os dados a serem processados.
    threshold : float, opcional (padrão=3.0)
        Limite de desvio padrão para considerar um valor como outlier.
    verbose : bool, opcional (padrão=True)
        Se True, imprime informações sobre a remoção de outliers.
        
    Retorna:
    --------
    pd.DataFrame
        DataFrame sem os outliers detectados.
        
    Exemplo:
    --------
    >>> df_clean = remove_outliers(df_raw, threshold=3.5)
    """
    # Seleciona apenas colunas numéricas
    numeric_cols = df.select_dtypes(include=[np.number])
    
    # Calcula Z-scores absolutos
    z_scores = np.abs((numeric_cols - numeric_cols.mean()) / numeric_cols.std())
    
    # Cria máscara para identificar linhas sem outliers
    mask = (z_scores < threshold).all(axis=1)
    
    # Gera relatório se verbose=True
    if verbose:
        n_outliers = len(df) - mask.sum()
        pct_outliers = n_outliers / len(df) * 100
        
        print("\n" + "="*50)
        print("RELATÓRIO DE REMOÇÃO DE OUTLIERS")
        print("-"*50)
        print(f"Total de linhas original: {len(df):,}")
        print(f"Outliers removidos: {n_outliers:,} ({pct_outliers:.2f}%)")
        print(f"Linhas restantes: {mask.sum():,}")
        print("="*50 + "\n")
    
    return df[mask].copy()

# =============================================================================
# B.1 Aplicando a Função para remover os outliers:
# =============================================================================
if __name__ == "__main__":
    # Demonstração da função
    print("ANTES DA REMOÇÃO:")
    print(f"Dimensões do dataset: {dataset2.shape}")
    print(f"Colunas: {list(dataset2.columns)}")
    print(dataset2.describe().round(2))
    
    # Aplica remoção de outliers
    dataset_clean = remove_outliers(
        df=dataset2,
        threshold=3,
        verbose=True
    )
    
    # Resultados
    print("\nAPÓS REMOÇÃO:")
    print(f"Dimensões do dataset: {dataset_clean.shape}")
    print("\nEstatísticas descritivas:")
    print(dataset_clean.describe().round(2))

    dataset2 = dataset_clean
    print(dataset2.head(2))
   

# =============================================================================
# C.Função para plotar matriz de correlação
# =============================================================================

def plot_correlation_matrix(df: pd.DataFrame, 
                          figsize: tuple = (16, 9), 
                          fontsize: int = 10,
                          cmap: str = 'coolwarm',
                          title: str = 'Matriz de Correlação entre Variáveis') -> None:
    """
    Plota um heatmap da matriz de correlação com formatação profissional.
    
    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame contendo os dados para cálculo de correlação.
    figsize : tuple, opcional (padrão=(16,9))
        Tamanho da figura (largura, altura).
    fontsize : int, opcional (padrão=10)
        Tamanho da fonte para as anotações.
    cmap : str, opcional (padrão='coolwarm')
        Mapa de cores para o heatmap.
    title : str, opcional (padrão='Matriz de Correlação entre Variáveis')
        Título do gráfico.
        
    Retorna:
    --------
    None (exibe o gráfico diretamente)
    """
    # Calcula a matriz de correlação
    corr = df.corr()
    
    # Cria máscara para o triângulo superior
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Configurações do plot
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        annot_kws={"size": fontsize},
        fmt=".2f",
        center=0,
        square=False,
        cbar_kws={"shrink": 1}
    )
    
    # Formatação dos eixos
    plt.xticks(rotation=90, ha='center')
    plt.yticks(rotation=0, va='center')
    
    # Título e layout
    plt.title(title, pad=20, fontsize=fontsize+4)
    plt.tight_layout()
    plt.show()


# =============================================================================
# C.1APLICAÇÃO NOS DATASETS
# =============================================================================

# Versão com personalização
plot_correlation_matrix(
    df=dataset1,
    figsize=(14, 10),
    fontsize=12,
    cmap='vlag',
    title='Correlação entre Variáveis Financeiras das Seguradoras')


plot_correlation_matrix(
    df=dataset1[colunas_calculadas],
    figsize=(14, 10),
    fontsize=12,
    cmap='vlag',
    title='Correlação entre Variáveis Financeiras das Seguradoras')


plot_correlation_matrix(
    df=dataset1[colunas_3_eixos],
    figsize=(14, 10),
    fontsize=12,
    cmap='vlag',
    title='Correlação entre Variáveis Financeiras das Seguradoras')

# =============================================================================
# D.Escalonamento das variáveis
# =============================================================================

# Padronização (crucial para PCA)
scaler = StandardScaler().fit(dataset2)
dataset2_escalonado = pd.DataFrame(scaler.fit_transform(dataset2),columns=dataset2.columns,index=dataset2.index)
#Resumir os dados reescalados
print(dataset2_escalonado.head(20))


sns.boxplot(data=dataset2_escalonado[['Índice de Solvência','Índice Combinado','Participação Relativa de Mercado']])
plt.xticks(rotation=45)
plt.show()

# =============================================================================
# E.Otimizando os Clusters
# =============================================================================


def find_optimal_clusters(data, max_k=15):
    """
    Determina o número ótimo de clusters usando método Elbow e Silhouette.
    
    Retorna:
        distortions (list): Valores de distorção para cada k
        silhouette_scores (list): Valores de silhouette para cada k
    """
    distortions = []
    silhouette_scores = []
    
    for k in range(2, max_k+1):
        kmeans = KMeans(n_clusters=k, n_init=10, init='k-means++', random_state=42)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)
        silhouette_scores.append(metrics.silhouette_score(data, kmeans.labels_))
    
    # Plot dos resultados
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.plot(range(2, max_k+1), distortions, 'bo-')
    ax1.set_title('Método Elbow')
    ax1.set_xlabel('Número de Clusters')
    ax1.grid(True)
    
    ax2.plot(range(2, max_k+1), silhouette_scores, 'go-')
    ax2.set_title('Score de Silhouette')
    ax2.set_xlabel('Número de Clusters')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return distortions, silhouette_scores

def plot_3d_clustering(data, labels, centroids, features, original_data=None, title='Agrupamento K-means'):
    """Visualização 3D interativa dos clusters"""
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    cores_clusters = ['#3357FF',"#B90606" ,'#33FF57',"#FF7A33",  "#F3EE69", "#077120"]
    scatter = ax.scatter(
        data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2],
        c=[cores_clusters[i] for i in labels],
         s=60, alpha=0.7, edgecolor='k'
    )
    if original_data is not None:
        
        tooltip_labels = [
            f"Seguradora: {original_data.iloc[i]['Grupo Econômico']}\n"
            f"Ramo: {original_data.iloc[i]['Ramo']}\n"
            f"Cluster: {labels[i]}"
            for i in range(len(data))
        ]
        cursor = mplcursors.cursor(scatter, hover=True)
        cursor.connect("add", lambda sel: sel.annotation.set_text(tooltip_labels[sel.index]))







    
    ax.scatter(
        centroids[:, 0], centroids[:, 1], centroids[:, 2],
        c='black', marker='.', s=100, alpha=1, label='Centróides'
    )
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_zlabel(features[2])
    plt.legend()
    
    plt.colorbar(scatter, pad=0.1).set_label('Cluster', rotation=270, labelpad=20)
    plt.tight_layout()
    plt.show()



# =============================================================================
# 4. MODELAGEM E AVALIAÇÃO
# =============================================================================
print("\n=== DETERMINAÇÃO DO NÚMERO DE CLUSTERS ===")
find_optimal_clusters(dataset2_escalonado, max_k=10)

# Definir número de clusters (baseado na análise)
N_CLUSTERS = 6

# Treinar modelo final
print(f"\n=== TREINANDO MODELO COM {N_CLUSTERS} CLUSTERS ===")
kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=10, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(dataset2_escalonado)

# =============================================================================
# 5. VISUALIZAÇÃO E ANÁLISE
# =============================================================================
# Obter centróides na escala original
centroids = scaler.inverse_transform(kmeans.cluster_centers_)

# Plot 3D
print("\n=== VISUALIZAÇÃO DOS CLUSTERS ===")
plot_3d_clustering(dataset2_escalonado, clusters, kmeans.cluster_centers_, colunas_3_eixos, 
                   original_data=dataset_completo, ## adicionado
                  title='Agrupamento de Seguradoras (K-means++)')

# Resultados
print("\n=== CENTRÓIDES FINAIS ===")
centroids_df = pd.DataFrame(centroids, columns=colunas_3_eixos)
print(centroids_df.round(2))

# Adicionar clusters aos dados originais
dataset2['cluster'] = clusters
dataset1['cluster'] = kmeans.fit_predict(dataset1)

print(dataset2.head(5))

# Estatísticas por cluster
print("\n=== ESTATÍSTICAS POR CLUSTER ===")
cluster_stats = dataset2.groupby('cluster')[colunas_3_eixos].agg(['mean', 'std', 'count'])
cluster_media = dataset2.groupby('cluster')[colunas_3_eixos].agg(['mean'])
print(cluster_stats.round(2))

# =============================================================================
# 6. INTERPRETAÇÃO DOS CLUSTERS
# =============================================================================

ax = cluster_media.plot.bar(
    rot=45,                  # Rótulos horizontais no eixo x
    figsize=(12, 6),        # Tamanho da figura (largura, altura)
    width=0.8,              # Largura das barras
    color=['#1f77b4', '#ff7f0e', '#2ca02c'],  # Cores para as 3 variáveis
    edgecolor='black',       # Borda preta nas barras
    alpha=0.7               # Transparência leve
)

# 3. Personalizar o gráfico
plt.title('Médias das Variáveis por Cluster', fontsize=14, pad=20)
plt.xlabel('Cluster', fontsize=12)
plt.ylabel('Médias', fontsize=12)
plt.legend(title='Variáveis', bbox_to_anchor=(1.05, 1), loc = 'upper left')  # Legenda fora do gráfico

# 4. Adicionar valores nas barras
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', 
                xytext=(0, 5), 
                textcoords='offset points')

plt.tight_layout()  # Ajusta o layout para evitar cortes
plt.show()

print("\n=== INTERPRETAÇÃO DOS CLUSTERS ===")
interpretation = {
    0: "Primeiro Cluster",
    1: "Segundo Cluster",
    2: "Terceiro Cluster",
    3: "Quarto Cluster",
    4:  "Quinto Cluster",
    5:  "Sexto Cluster"
} 

for cluster_num, description in interpretation.items():
    print(f"Cluster {cluster_num}: {description}")
    print(f"Perfil médio: {centroids_df.iloc[cluster_num].round(2).to_dict()}\n")


print(dataset_completo.head(5))







dataset2['Cluster2'] = kmeans.fit_predict(dataset2_escalonado)
cluster_summary = dataset2.groupby('Cluster2')[colunas_3_eixos].mean().reset_index()

fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')

# Cores para cada cluster
""" colors = ['lightblue', 'lightgreen', 'salmon', 'gold','red', 'purple']
cell_colors = [['white'] * len(cluster_summary.columns)] + [
    [colors[i]] * len(cluster_summary.columns) for i in range(len(cluster_summary))
] """


table = ax.table(
    cellText=cluster_summary.round(2).values,
    colLabels=cluster_summary.columns,
    loc='center'
    
)

table.set_fontsize(12)
table.scale(1.2, 1.5)
plt.title("Médias por Cluster", pad=20)
plt.show()



# =============================================================================
# 7. Aplicação com novos dados:
# =============================================================================

#novos dados de exemplo:

novo_ponto = np.array([[1, 0.8, 0.5]]) # Índice de solvência = 3,5 Índice de solvência = 0.45 e participação relativa =1.2
novo_ponto_escalado = scaler.transform(novo_ponto)
cluster_novo_ponto = kmeans.predict(novo_ponto_escalado)[0]
print(f"O novo ponto pertence ao Cluster: {cluster_novo_ponto}")




