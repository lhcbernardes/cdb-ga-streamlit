import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import random
from datetime import datetime
from io import BytesIO
import requests

# Configuração da página
st.set_page_config(page_title="GA Tesouro Direto Otimizador", layout="wide")
st.title("📈 Otimização de Portfólio - Tesouro Direto")

# Injetar CSS para limitar o tamanho das figuras
st.markdown(
    """
    <style>
    .stImage > img, .element-container img {
        max-width: 300px !important;
        width: 100%;
        height: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Explicação do cálculo do score
with st.expander("ℹ️ Como o Score é Calculado"):
    st.markdown("""
    O **score** é uma pontuação que avalia o desempenho de cada portfólio de títulos do Tesouro Direto, com base na estratégia escolhida. Cada estratégia foca em um aspecto diferente do investimento, como retorno, prazo, diversificação ou risco. Veja como cada uma funciona:

    - **Média da Rentabilidade**  
      Calcula a média simples das taxas de rentabilidade anual dos títulos selecionados no portfólio. É uma abordagem direta para avaliar o retorno esperado, sem considerar prazos ou riscos.  
      *Exemplo*: Se o portfólio tem três títulos com rentabilidades de 10%, 12% e 14%, o score será (10 + 12 + 14) / 3 = 12%.  
      *Ideal para*: Investidores que buscam simplicidade e priorizam o retorno médio.

    - **Rentabilidade Total até o Vencimento**  
      Considera a rentabilidade composta de cada título até sua data de vencimento, ajustada pelo prazo em anos. Essa estratégia reflete o retorno acumulado que você teria se mantivesse os títulos até o fim.  
      *Exemplo*: Um título com rentabilidade de 10% ao ano e vencimento em 2 anos teria um retorno composto de (1 + 0.10)² - 1 = 21%. O score é a média desses retornos para o portfólio.  
      *Ideal para*: Investidores focados no retorno de longo prazo, considerando o efeito dos juros compostos.

    - **Rentabilidade Ajustada pelo Prazo**  
      Calcula a média das rentabilidades, mas aplica uma penalidade baseada no prazo médio dos títulos (em dias). Títulos com prazos mais longos recebem uma pequena redução no score, refletindo o risco de manter o investimento por mais tempo.  
      *Exemplo*: Um portfólio com rentabilidade média de 12% e prazo médio de 730 dias (2 anos) recebe uma penalidade de 0.005 * 730 / 365 = 0.01 (1%). O score seria 12% - 1% = 11%.  
      *Ideal para*: Investidores que preferem equilibrar retorno e liquidez, evitando prazos muito longos.

    - **Diversificação de Tipos**  
      Calcula a média das rentabilidades e adiciona um bônus proporcional ao número de tipos diferentes de títulos no portfólio (como Tesouro Selic, IPCA+, etc.). Isso incentiva a diversificação para reduzir riscos.  
      *Exemplo*: Um portfólio com rentabilidade média de 12% e 3 tipos diferentes de títulos ganha um bônus de 0.5 * 3 = 1.5%, resultando em um score de 12% + 1.5% = 13.5%.  
      *Ideal para*: Investidores que valorizam a diversificação para maior estabilidade.

    - **Índice Sharpe Simplificado**  
      Mede a relação entre o retorno excedente (rentabilidade média menos a taxa livre de risco, como a Selic) e o risco (desvio padrão das rentabilidades dos títulos). Um score maior indica melhor retorno ajustado ao risco.  
      *Exemplo*: Um portfólio com rentabilidade média de 12%, desvio padrão de 2% e taxa livre de risco de 10% tem um score de (12 - 10) / 2 = 1.0.  
      *Ideal para*: Investidores que buscam otimizar o retorno considerando o risco envolvido.
    """)

# Função para carregar e processar dados
@st.cache_data
def carregar_dados():
    try:
        url = "https://www.tesourotransparente.gov.br/ckan/dataset/df56aa42-484a-4a59-8184-7676580c81e3/resource/796d2059-14e9-44e3-80c9-2d9e30b405c1/download/precotaxatesourodireto.csv"
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
        content = BytesIO()
        for chunk in r.iter_content(1024):
            content.write(chunk)
        content.seek(0)
        df = pd.read_csv(content, sep=";", encoding="utf-8")
        
        # Verificar colunas obrigatórias
        required_columns = ["Data Vencimento", "Data Base", "Taxa Compra Manha", "Tipo Titulo"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise KeyError(f"Colunas ausentes no CSV: {missing_columns}")
        
        # Processar datas
        df["Data Vencimento"] = pd.to_datetime(df["Data Vencimento"], dayfirst=True, errors="coerce")
        df["Data Base"] = pd.to_datetime(df["Data Base"], dayfirst=True, errors="coerce")
        df["Rentabilidade"] = df["Taxa Compra Manha"].str.replace(",", ".").astype(float)
        df["Prazo"] = (df["Data Vencimento"] - df["Data Base"]).dt.days
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None

# Carregar dados
raw_df = carregar_dados()
if raw_df is None:
    st.stop()

# Filtrar títulos com vencimento futuro
hoje = pd.to_datetime(datetime.now().date())
raw_df = raw_df[raw_df["Data Vencimento"] > hoje].copy()

st.success(f"{len(raw_df)} títulos com vencimento futuro carregados.")
st.dataframe(raw_df.head())

# Parâmetros do algoritmo
st.sidebar.header("⚙️ Parâmetros do Algoritmo")
POP_SIZE = st.sidebar.slider("Tamanho da População", 50, 500, 100, step=50)
NGEN = st.sidebar.slider("Número de Gerações", 10, 500, 100, step=10)
CXPB = st.sidebar.slider("Probabilidade de Crossover", 0.5, 1.0, 0.7, step=0.05)
MUTPB = st.sidebar.slider("Probabilidade de Mutação", 0.1, 0.9, 0.5, step=0.05)
N_ATIVOS = st.sidebar.slider("Títulos por Portfólio", 3, min(10, len(raw_df)), 5)
risk_free_rate = st.sidebar.slider("Taxa Livre de Risco (%)", 5.0, 15.0, 10.0, step=0.5)
estrategia = st.sidebar.selectbox("Estratégia de Score", [
    "Média da Rentabilidade",
    "Rentabilidade Total até o Vencimento",
    "Rentabilidade Ajustada pelo Prazo",
    "Diversificação de Tipos",
    "Índice Sharpe Simplificado"
])

# Validação de parâmetros
if len(raw_df) < N_ATIVOS:
    st.error(f"Quantidade de títulos disponíveis ({len(raw_df)}) menor que o número de títulos por portfólio ({N_ATIVOS}).")
    st.stop()

# Configurações do DEAP
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Funções do algoritmo genético
def gerador_indices():
    return random.sample(range(len(raw_df)), N_ATIVOS)

toolbox.register("indices", gerador_indices)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)

def mutShuffle(ind, indpb):
    if random.random() < indpb:
        # Criar uma cópia do indivíduo como lista
        ind_copy = list(ind)
        random.shuffle(ind_copy)
        # Aplicar reparo
        ind_copy = repair_individual(ind_copy)
        # Criar um novo indivíduo com os índices reparados
        new_ind = creator.Individual(ind_copy)
        # Copiar o fitness do indivíduo original, se existir
        if hasattr(ind, 'fitness') and ind.fitness.valid:
            new_ind.fitness.values = ind.fitness.values
        return new_ind,
    return ind,

toolbox.register("mutate", mutShuffle, indpb=0.5)
toolbox.register("select", tools.selTournament, tournsize=3)

def repair_individual(ind):
    unique_ind = list(dict.fromkeys(ind))  # Remove duplicatas mantendo ordem
    attempts = 0
    max_attempts = 100
    while len(unique_ind) < N_ATIVOS and attempts < max_attempts:
        new_idx = random.randint(0, len(raw_df) - 1)
        if new_idx not in unique_ind:
            unique_ind.append(new_idx)
        attempts += 1
    return unique_ind[:N_ATIVOS]

def evaluate(ind):
    try:
        selected = raw_df.iloc[ind]
        if estrategia == "Média da Rentabilidade":
            return (selected["Rentabilidade"].mean(),)
        elif estrategia == "Rentabilidade Total até o Vencimento":
            anos = selected["Prazo"] / 365
            total = ((1 + selected["Rentabilidade"] / 100) ** anos - 1).mean()
            return (total * 100,)
        elif estrategia == "Rentabilidade Ajustada pelo Prazo":
            penalidade = 0.005 * selected["Prazo"].mean() / 365
            return (selected["Rentabilidade"].mean() - penalidade,)
        elif estrategia == "Diversificação de Tipos":
            n_tipos = selected["Tipo Titulo"].nunique()
            return (selected["Rentabilidade"].mean() + 0.5 * n_tipos,)
        elif estrategia == "Índice Sharpe Simplificado":
            media = selected["Rentabilidade"].mean()
            desvio = selected["Rentabilidade"].std() or 1e-6
            return ((media - risk_free_rate) / desvio,)
        return (0.0,)
    except Exception as e:
        st.warning(f"Erro na avaliação: {e}")
        return (0.0,)

toolbox.register("evaluate", evaluate)

# Função para plotar evolução
def plot_evolution(log):
    geracoes, melhores, medias = zip(*log)
    fig, ax = plt.subplots(figsize=(3, 2), dpi=80)  # Tamanho reduzido
    ax.plot(geracoes, melhores, label='Melhor Score')
    ax.plot(geracoes, medias, label='Média da População', linestyle='--')
    ax.set_title("Evolução do Score", fontsize=8)
    ax.set_xlabel("Geração", fontsize=6)
    ax.set_ylabel("Score", fontsize=6)
    ax.legend(fontsize=6)
    ax.tick_params(axis='both', which='major', labelsize=6)
    plt.tight_layout()
    return fig

# Função principal de execução
@st.cache_data
def rodar_otimizacao(_pop_size, _ngen, _cxpb, _mutpb, _n_ativos, _estrategia, _risk_free_rate):
    pop = toolbox.population(n=_pop_size)
    for ind in pop:
        if not isinstance(ind, creator.Individual):
            st.error(f"Indivíduo inválido na inicialização: {type(ind)}")
            return None, []
        ind.fitness.values = toolbox.evaluate(ind)
    
    log = []
    for g in range(_ngen):
        offspring = algorithms.varAnd(pop, toolbox, cxpb=_cxpb, mutpb=_mutpb)
        for ind in offspring:
            if not isinstance(ind, creator.Individual):
                st.error(f"Indivíduo inválido após varAnd: {type(ind)}")
                return None, log
            ind[:] = repair_individual(ind)
            ind.fitness.values = toolbox.evaluate(ind)
        pop = toolbox.select(pop + offspring, k=_pop_size)
        melhor = tools.selBest(pop, k=1)[0]
        avg_score = np.mean([i.fitness.values[0] for i in pop])
        log.append((g + 1, melhor.fitness.values[0], avg_score))
    
    return pop, log

# Botão para executar otimização
if st.button("🚀 Rodar Otimização"):
    with st.spinner("Executando algoritmo genético..."):
        pop, log = rodar_otimizacao(POP_SIZE, NGEN, CXPB, MUTPB, N_ATIVOS, estrategia, risk_free_rate)
        
        if pop is None:
            st.error("Erro na execução do algoritmo genético. Verifique os logs acima.")
            st.stop()
        
        # Plotar evolução
        grafico_area = st.empty()
        fig = plot_evolution(log)
        grafico_area.pyplot(fig, use_container_width=False)
        plt.close(fig)
        
        # Melhor solução
        melhor = tools.selBest(pop, k=1)[0]
        resultado = raw_df.iloc[melhor].copy()
        resultado["Score"] = melhor.fitness.values[0]
        
        st.markdown("### 🏆 Portfólio Ideal")
        st.dataframe(resultado)
        
        # Detalhes do resultado
        with st.expander("📊 Detalhes do Resultado"):
            st.markdown(f"""
            - **Score final**: `{melhor.fitness.values[0]:.4f}`
            - **Estratégia usada**: `{estrategia}`
            - **Quantidade de ativos**: `{N_ATIVOS}`
            - **Melhor rentabilidade**: `{resultado["Rentabilidade"].mean():.2f}%`
            - **Prazo médio (dias)**: `{resultado["Prazo"].mean():.0f}`
            - **Tipos de títulos diferentes**: `{resultado["Tipo Titulo"].nunique()} ({", ".join(resultado["Tipo Titulo"].unique())})`
            """)
        
        # Gráfico de dispersão
        with st.expander("📈 Dispersão de Rendimento vs Retorno"):
            fig, ax = plt.subplots(figsize=(3, 2), dpi=80)  # Tamanho reduzido
            riscos = [raw_df.iloc[ind]["Rentabilidade"].std() or 0 for ind in pop]
            retornos = [raw_df.iloc[ind]["Rentabilidade"].mean() for ind in pop]
            ax.scatter(riscos, retornos, alpha=0.6)
            ax.set_xlabel("Dispersão de Yields (std, risco)", fontsize=6)
            ax.set_ylabel("Retorno Médio (%)", fontsize=6)
            ax.set_title("Dispersão vs Retorno nos Portfólios", fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=6)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)