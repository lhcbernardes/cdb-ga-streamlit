import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import random
from datetime import datetime
from io import BytesIO
import requests
import time
import copy

# Configuração da página com layout wide e ícone
st.set_page_config(page_title="GA Tesouro Direto Otimizador", layout="wide", page_icon="📈")

# Injetar CSS personalizado para melhores efeitos visuais (cores, botões, etc.)
st.markdown(
    """
    <style>
    .stButton > button {
        background-color: #4CAF50; /* Verde atraente */
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 12px;
        transition: background-color 0.3s;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .stSpinner > div {
        color: #4CAF50;
    }
    .stAlert {
        background-color: #f0f2f6;
        border-radius: 8px;
    }
    .element-container .stMarkdown {
        font-family: 'Arial', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📈 Otimização de Portfólio - Tesouro Direto")

# Expander com explicação resumida e visualmente atraente
with st.expander("ℹ️ Como o Score é Calculado", expanded=False):
    st.markdown("""
    O **score** avalia seu portfólio de Tesouro Direto com base em diferentes estratégias. Escolha uma e otimize!

    - **Média da Rentabilidade** 📊: Média simples das taxas anuais. Ideal para simplicidade.
    - **Rentabilidade Total até o Vencimento** ⏳: Retorno composto total, considerando prazos.
    - **Rentabilidade Ajustada pelo Prazo** ⚖️: Penaliza prazos longos para equilibrar liquidez.
    - **Diversificação de Tipos** 🌟: Bônus por variedade de títulos, reduzindo riscos.
    - **Sharpe Ratio** 📈: Retorno ajustado pelo risco (novo!).
    - **Multi-Objetivo** 🎯: Otimiza retorno, risco e diversificação simultaneamente.
    """)

# Função para carregar dados com cache para eficiência
@st.cache_data(ttl=3600)  # Cache por 1 hora
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
        
        # Processamento de dados
        df["Data Vencimento"] = pd.to_datetime(df["Data Vencimento"], dayfirst=True, errors="coerce")
        df["Data Base"] = pd.to_datetime(df["Data Base"], dayfirst=True, errors="coerce")
        df["Rentabilidade"] = df["Taxa Compra Manha"].str.replace(",", ".").astype(float)
        df["Prazo"] = (df["Data Vencimento"] - df["Data Base"]).dt.days
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return pd.DataFrame()

# Carregar dados com spinner visual
with st.spinner("📥 Baixando dados do Tesouro Direto..."):
    raw_df = carregar_dados()

# Filtrar títulos futuros
raw_df = raw_df[raw_df["Data Vencimento"] > datetime.now()].copy()

if raw_df.empty:
    st.error("Nenhum título com vencimento futuro disponível. Tente novamente mais tarde.")
    st.stop()

st.success(f"✅ {len(raw_df)} títulos com vencimento futuro carregados.")

# Exibir prévia dos dados em um expander para não poluir a tela
with st.expander("📋 Prévia dos Dados", expanded=False):
    st.dataframe(raw_df.head())

# Sidebar com parâmetros, organizado visualmente
st.sidebar.header("⚙️ Parâmetros do Algoritmo")
POP_SIZE = st.sidebar.slider("Tamanho da População", 50, 500, 100, step=50, help="Número de portfólios iniciais.")
NGEN = st.sidebar.slider("Máximo de Gerações", 10, 500, 100, step=10, help="Quantas iterações o algoritmo fará.")
CXPB = st.sidebar.slider("Probabilidade de Crossover", 0.5, 1.0, 0.7, step=0.05, help="Chance de combinar portfólios.")
MUTPB = st.sidebar.slider("Probabilidade de Mutação", 0.5, 1.0, 0.9, step=0.05, help="Chance de alterar portfólios.")
N_ATIVOS = st.sidebar.slider("Títulos por Portfólio", 3, min(10, len(raw_df)), 5, help="Quantos títulos em cada portfólio.")
estrategia = st.sidebar.selectbox("Estratégia de Score", [
    "Média da Rentabilidade",
    "Rentabilidade Total até o Vencimento",
    "Rentabilidade Ajustada pelo Prazo",
    "Diversificação de Tipos",
    "Sharpe Ratio",
    "Multi-Objetivo"
], help="Escolha como calcular o score.")

# Parâmetros avançados
st.sidebar.header("🔧 Parâmetros Avançados")
ELITE_SIZE = st.sidebar.slider("Tamanho da Elite (%)", 5, 20, 10, step=5, help="Percentual dos melhores indivíduos a preservar.")
TOURNAMENT_SIZE = st.sidebar.slider("Tamanho do Torneio", 2, 8, 4, step=1, help="Tamanho do torneio para seleção.")
DIVERSITY_THRESHOLD = st.sidebar.slider("Limiar de Diversidade", 0.1, 0.9, 0.3, step=0.1, help="Limiar para reinicialização por diversidade.")

# Validação de parâmetros
if len(raw_df) < N_ATIVOS:
    st.error(f"❌ Quantidade de títulos disponíveis ({len(raw_df)}) é menor que o número por portfólio ({N_ATIVOS}). Ajuste os parâmetros.")
    st.stop()

# Configurações do DEAP - CORREÇÃO DOS ERROS DE LINTER
try:
    del creator.FitnessMax  # type: ignore
except:
    pass
try:
    del creator.Individual  # type: ignore
except:
    pass

# type: ignore - DEAP cria atributos dinamicamente
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # type: ignore
creator.create("Individual", list, fitness=creator.FitnessMax)  # type: ignore

# Funções auxiliares melhoradas
def gerar_indices():
    """Gera índices únicos para representar um portfólio"""
    return random.sample(range(len(raw_df)), N_ATIVOS)

def repair(ind):
    """Repara indivíduo removendo duplicatas e garantindo tamanho correto"""
    unique = list(dict.fromkeys(ind))
    while len(unique) < N_ATIVOS:
        novo = random.randint(0, len(raw_df) - 1)
        if novo not in unique:
            unique.append(novo)
    return unique[:N_ATIVOS]

def calcular_diversidade(populacao):
    """Calcula a diversidade da população baseada na distância média entre indivíduos"""
    if len(populacao) < 2:
        return 0.0
    
    distancias = []
    for i in range(len(populacao)):
        for j in range(i+1, len(populacao)):
            # Distância baseada na sobreposição de títulos
            overlap = len(set(populacao[i]) & set(populacao[j])) / N_ATIVOS
            distancias.append(1 - overlap)
    
    return np.mean(distancias) if distancias else 0.0

# Operadores genéticos melhorados
def crossover_uniforme(ind1, ind2):
    """Crossover uniforme melhorado para portfólios"""
    child1 = []
    child2 = []
    
    # Usar máscara aleatória para decidir de qual pai pegar cada posição
    mask = [random.random() < 0.5 for _ in range(N_ATIVOS)]
    
    for i in range(N_ATIVOS):
        if mask[i]:
            child1.append(ind1[i])
            child2.append(ind2[i])
        else:
            child1.append(ind2[i])
            child2.append(ind1[i])
    
    # Reparar duplicatas
    child1 = repair(child1)
    child2 = repair(child2)
    
    return creator.Individual(child1), creator.Individual(child2)  # type: ignore

def mutacao_inteligente(ind, indpb=0.3):
    """Mutação inteligente que preserva alguns títulos bons"""
    if random.random() < indpb:
        ind_copy = list(ind)
        
        # Mutação por substituição parcial
        num_mutations = max(1, int(N_ATIVOS * 0.3))  # 30% dos títulos
        positions = random.sample(range(N_ATIVOS), num_mutations)
        
        for pos in positions:
            # Substituir por um título aleatório
            novo_titulo = random.randint(0, len(raw_df) - 1)
            while novo_titulo in ind_copy:
                novo_titulo = random.randint(0, len(raw_df) - 1)
            ind_copy[pos] = novo_titulo
        
        return creator.Individual(ind_copy),  # type: ignore
    return ind,

def mutacao_swap(ind, indpb=0.2):
    """Mutação por troca de posições"""
    if random.random() < indpb:
        ind_copy = list(ind)
        # Trocar duas posições aleatórias
        pos1, pos2 = random.sample(range(N_ATIVOS), 2)
        ind_copy[pos1], ind_copy[pos2] = ind_copy[pos2], ind_copy[pos1]
        return creator.Individual(ind_copy),  # type: ignore
    return ind,

def evaluate(ind):
    """Função de avaliação melhorada com múltiplas estratégias"""
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
            tipos = selected["Tipo Titulo"].nunique()
            return (selected["Rentabilidade"].mean() + 0.5 * tipos,)
        
        elif estrategia == "Sharpe Ratio":
            retorno = selected["Rentabilidade"].mean()
            risco = selected["Rentabilidade"].std() or 0.1  # Evitar divisão por zero
            sharpe = retorno / risco if risco > 0 else 0
            return (sharpe,)
        
        elif estrategia == "Multi-Objetivo":
            # Retorno, risco e diversificação
            retorno = selected["Rentabilidade"].mean()
            risco = selected["Rentabilidade"].std() or 0.1
            diversidade = selected["Tipo Titulo"].nunique()
            # Score composto
            score = retorno - 0.5 * risco + 0.3 * diversidade
            return (score,)
        
        return (0.0,)
    except Exception as e:
        st.warning(f"Erro na avaliação: {e}")
        return (0.0,)

# Cria toolbox e registrar funções melhoradas
toolbox = base.Toolbox()
toolbox.register("indices", gerar_indices)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)  # type: ignore
toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # type: ignore
toolbox.register("mate", crossover_uniforme)
toolbox.register("mutate", mutacao_inteligente)
toolbox.register("mutate_swap", mutacao_swap)
toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)
toolbox.register("evaluate", evaluate)

# Função para plotar evolução com tema visual melhorado
def plot_evolucao(log):
    gens, melhores, medias = zip(*log) if log else ([], [], [])
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.style.use('seaborn-v0_8')
    ax.plot(gens, melhores, label="Melhor Score", color='green', linewidth=2)
    ax.plot(gens, medias, label="Média da População", color='blue', linestyle='--', linewidth=1.5)
    ax.set_title("Evolução da Otimização", fontsize=12)
    ax.set_xlabel("Geração", fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig

# Função principal de otimização melhorada
def rodar_otimizacao():
    """Algoritmo genético melhorado com diversidade e early stopping"""
    # Seed fixo para reproducibilidade
    random.seed(42)
    
    # Inicializar população
    pop = toolbox.population(n=POP_SIZE)  # type: ignore
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)  # type: ignore

    # Placeholder para gráfico e barra de progresso
    grafico_area = st.empty()
    progress_bar = st.progress(0)
    log = []
    best_score = -np.inf
    no_improvement = 0
    early_stop_limit = 20  # Aumentado para dar mais chance
    elite_size = max(1, int(POP_SIZE * ELITE_SIZE / 100))

    for g in range(1, NGEN + 1):
        # Gerar offspring
        offspring = algorithms.varAnd(pop, toolbox, cxpb=CXPB, mutpb=MUTPB)
        
        # Reparar e avaliar offspring
        for ind in offspring:
            ind[:] = repair(ind)
            ind.fitness.values = toolbox.evaluate(ind)  # type: ignore

        # Elitismo melhorado
        elite = tools.selBest(pop, k=elite_size)
        pop = toolbox.select(pop + offspring, k=POP_SIZE - len(elite)) + elite  # type: ignore

        # Calcular métricas
        melhor = tools.selBest(pop, k=1)[0]
        media = np.mean([i.fitness.values[0] for i in pop if i.fitness.valid])
        log.append((g, melhor.fitness.values[0], media))

        # Atualizar gráfico em tempo real
        fig = plot_evolucao(log)
        grafico_area.pyplot(fig)
        plt.close(fig)

        # Atualizar progresso
        progress_bar.progress(g / NGEN)

        # Early stopping melhorado
        if melhor.fitness.values[0] > best_score:
            best_score = melhor.fitness.values[0]
            no_improvement = 0
        else:
            no_improvement += 1
        
        # Verificar diversidade
        diversidade = calcular_diversidade(pop)
        if diversidade < DIVERSITY_THRESHOLD and g > 10:
            # Reinicializar parte da população para manter diversidade
            num_reinit = int(POP_SIZE * 0.2)
            for _ in range(num_reinit):
                novo_ind = toolbox.individual()  # type: ignore
                novo_ind.fitness.values = toolbox.evaluate(novo_ind)  # type: ignore
                pop[random.randint(0, len(pop)-1)] = novo_ind
            st.info(f"🔄 Reinicialização por diversidade na geração {g}")
        
        if no_improvement >= early_stop_limit:
            st.info(f"🛑 Otimização parou na geração {g} devido a estagnação (sem melhorias).")
            break
        
        # Pequeno delay para efeito "tempo real"
        time.sleep(0.05)  # Reduzido para melhor performance

    return pop, log

# Botão para rodar com estilo visual
if st.button("🚀 Rodar Otimização", help="Inicie a otimização com os parâmetros selecionados."):
    with st.spinner("🔄 Otimizando portfólio... Aguarde!"):
        pop, log = rodar_otimizacao()

        if not pop:
            st.error("❌ Erro na otimização. Verifique os parâmetros e tente novamente.")
            st.stop()

        # Exibir resultados em colunas para melhor layout
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🏆 Melhor Portfólio Encontrado")
            melhor = tools.selBest(pop, k=1)[0]
            resultado = raw_df.iloc[melhor].copy()
            resultado["Score"] = melhor.fitness.values[0]
            st.dataframe(resultado.style.format({"Rentabilidade": "{:.2f}%", "Prazo": "{:.0f} dias"}))

        with col2:
            st.subheader("📊 Detalhes do Melhor Portfólio")
            st.markdown(f"""
            - **Score Final**: {melhor.fitness.values[0]:.4f}
            - **Estratégia**: {estrategia}
            - **Rentabilidade Média**: {resultado["Rentabilidade"].mean():.2f}%
            - **Prazo Médio**: {resultado["Prazo"].mean():.0f} dias
            - **Diversidade de Títulos**: {resultado["Tipo Titulo"].nunique()}
            - **Risco (Desvio Padrão)**: {resultado["Rentabilidade"].std():.2f}%
            """)

        # Gráfico de Pareto em expander
        with st.expander("📈 Gráfico de Pareto (Risco vs. Retorno)", expanded=True):
            fig, ax = plt.subplots(figsize=(6, 4))
            plt.style.use('seaborn-v0_8')
            riscos = []
            retornos = []
            for ind in pop:
                dados = raw_df.iloc[ind]
                riscos.append(dados["Rentabilidade"].std() or 0)
                retornos.append(dados["Rentabilidade"].mean())

            ax.scatter(riscos, retornos, alpha=0.6, color='blue', label='Portfólios')
            # Destacar o melhor
            melhor_dados = raw_df.iloc[melhor]
            ax.scatter(melhor_dados["Rentabilidade"].std() or 0, melhor_dados["Rentabilidade"].mean(), color='red', s=100, label='Melhor Portfólio')
            ax.set_title("Fronteira de Pareto", fontsize=12)
            ax.set_xlabel("Risco (Desvio Padrão)", fontsize=10)
            ax.set_ylabel("Retorno Médio (%)", fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.tick_params(labelsize=8)
            fig.tight_layout()
            st.pyplot(fig)

        # Análise de diversidade da população final
        with st.expander("🌐 Análise de Diversidade da População", expanded=False):
            diversidade_final = calcular_diversidade(pop)
            st.metric("Diversidade da População Final", f"{diversidade_final:.3f}")
            
            # Histograma de scores
            scores = [ind.fitness.values[0] for ind in pop]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(scores, bins=20, alpha=0.7, color='green')
            ax.axvline(float(np.mean(scores)), color='red', linestyle='--', label='Média')
            ax.set_title("Distribuição de Scores da População Final")
            ax.set_xlabel("Score")
            ax.set_ylabel("Frequência")
            ax.legend()
            st.pyplot(fig)
