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

st.set_page_config(page_title="GA Tesouro Direto Otimizador", layout="wide")
st.title("📈 Otimização dos Melhores Títulos do Tesouro Direto")

# Função de download com cache
@st.cache_data(show_spinner=False)
def baixar_arquivo_tesouro():
    TESOURO_URL = "https://www.tesourotransparente.gov.br/ckan/dataset/df56aa42-484a-4a59-8184-7676580c81e3/resource/796d2059-14e9-44e3-80c9-2d9e30b405c1/download/precotaxatesourodireto.csv"
    FILE_SIZE = 13_119_488  # bytes esperados
    response = requests.get(TESOURO_URL, stream=True)
    response.raise_for_status()
    total_bytes = 0
    content = b""
    progress_bar = st.progress(0)
    status_text = st.empty()

    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            content += chunk
            total_bytes += len(chunk)
            progresso = min(total_bytes / FILE_SIZE, 1.0)
            progress_bar.progress(progresso)
            status_text.text(f"📊 Baixado: {total_bytes / 1_048_576:.2f} MB / {FILE_SIZE / 1_048_576:.2f} MB")

    return content

# Download do CSV do Tesouro Direto
st.markdown("### 📥 Baixando dados do Tesouro Direto...")
conteudo_binario = baixar_arquivo_tesouro()
uploaded_file = BytesIO(conteudo_binario)

# Leitura e tratamento
try:
    raw_df = pd.read_csv(uploaded_file, sep=";", encoding="utf-8")
    venc = pd.to_datetime(raw_df["Data Vencimento"], dayfirst=True, errors='coerce')
    data_base = pd.to_datetime(raw_df["Data Base"], dayfirst=True, errors='coerce')
    hoje = pd.Timestamp(datetime.today().date())
    validos = venc > hoje
    raw_df = raw_df[validos]
    venc = venc[validos]
    data_base = data_base[validos]
    tesouros = pd.DataFrame()
    tesouros["Banco"] = raw_df["Tipo Titulo"]
    tesouros["Rentabilidade"] = raw_df["Taxa Compra Manha"].str.replace(",", ".", regex=False).astype(float)
    tesouros["Prazo"] = (venc - data_base).dt.days
    tesouros["Rating"] = "A+"
except Exception as e:
    st.error(f"Erro ao processar CSV: {e}")
    st.stop()

st.success(f"✅ {len(tesouros)} títulos com vencimento futuro carregados com sucesso.")
st.dataframe(tesouros.head())

# Parâmetros do algoritmo
st.sidebar.header("⚙️ Parâmetros do Algoritmo Genético")
POP_SIZE = st.sidebar.slider("Tamanho da população", 50, 1000, 200, 50)
NGEN = st.sidebar.slider("Número de gerações", 10, 1000, 300, 10)
CXPB = st.sidebar.slider("Probabilidade de crossover", 0.5, 1.0, 0.7, 0.05)
MUTPB = st.sidebar.slider("Probabilidade de mutação", 0.1, 0.8, 0.4, 0.05)
N_ATIVOS = st.sidebar.slider("Títulos por portfólio", 3, 10, 5)

# Filtros
st.sidebar.header("🔍 Filtros de Qualidade")
bancos_excluidos = st.sidebar.multiselect("Excluir títulos de", sorted(tesouros["Banco"].unique()))
rating_opcional = st.sidebar.toggle("Filtrar por Rating", value=True)
ratings_aceitos = []
if rating_opcional:
    ratings_aceitos = st.sidebar.multiselect("Ratings aceitos", sorted(tesouros["Rating"].unique()), default=sorted(tesouros["Rating"].unique()))

filtro = pd.Series([True] * len(tesouros), index=tesouros.index)
if bancos_excluidos:
    filtro &= ~tesouros["Banco"].isin(bancos_excluidos)
if rating_opcional and ratings_aceitos:
    filtro &= tesouros["Rating"].isin(ratings_aceitos)
tesouros = tesouros[filtro]
st.write("📊 Títulos disponíveis após filtros:", len(tesouros))
if len(tesouros) < N_ATIVOS:
    st.error("❌ Poucos títulos após os filtros para otimização. Reduza os filtros ou N_ATIVOS.")
    st.stop()

# Funções do algoritmo
def evaluate(individual):
    selected = tesouros.iloc[individual]
    retorno = selected["Rentabilidade"].mean()
    return (retorno,)

def repair_individual(individual):
    seen = set()
    unique = []
    duplicates = []
    for gene in individual:
        if gene in seen:
            duplicates.append(gene)
        else:
            seen.add(gene)
            unique.append(gene)
    available = list(set(range(len(tesouros))) - set(unique))
    random.shuffle(available)
    while len(unique) < len(individual) and available:
        unique.append(available.pop())
    return creator.Individual(unique)

# DEAP setup
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("indices", lambda: random.sample(range(len(tesouros)), N_ATIVOS))
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.3)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# Execução
if st.button("🚀 Rodar otimização"):
    start_time = time.time()
    with st.spinner("Executando..."):
        population = toolbox.population(n=POP_SIZE)
        for ind in population:
            ind.fitness.values = toolbox.evaluate(ind)

        st.markdown("### 📉 Evolução em Tempo Real")
        chart_data = pd.DataFrame(columns=["Melhor Score", "Média Score"])
        chart = st.line_chart(chart_data)

        log = []
        best_score = -np.inf
        rounds_without_improvement = 0
        early_stopping_rounds = 50

        for gen in range(1, NGEN + 1):
            offspring = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)
            offspring = [repair_individual(ind) for ind in offspring]
            for ind in offspring:
                ind.fitness.values = toolbox.evaluate(ind)
            population = toolbox.select(population + offspring, k=POP_SIZE)

            best = tools.selBest(population, k=1)[0]
            avg_fit = np.mean([ind.fitness.values[0] for ind in population])
            log.append((gen, best.fitness.values[0], avg_fit))

            # Atualiza gráfico dinâmico
            new_row = pd.DataFrame([[best.fitness.values[0], avg_fit]], columns=["Melhor Score", "Média Score"])
            chart.add_rows(new_row)

            # Early stopping
            if best.fitness.values[0] > best_score:
                best_score = best.fitness.values[0]
                rounds_without_improvement = 0
            else:
                rounds_without_improvement += 1
                if rounds_without_improvement >= early_stopping_rounds:
                    st.warning(f"⏹️ Parado antecipadamente na geração {gen} após {early_stopping_rounds} gerações sem melhoria.")
                    break

        tempo_total = time.time() - start_time
        minutos = int(tempo_total // 60)
        segundos = int(tempo_total % 60)
        st.success(f"✅ Finalizado em {minutos}m {segundos}s")

        last_gen = log[-1]
        st.info(f"Geração {last_gen[0]}: Melhor Score = {last_gen[1]:.2f} | Média = {last_gen[2]:.2f}")

        generations, best_scores, avg_scores = zip(*log)
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.plot(generations, best_scores, label="Melhor Score")
        ax.plot(generations, avg_scores, label="Média Score", linestyle="--")
        ax.set_xlabel("Geração")
        ax.set_ylabel("Score")
        ax.set_title("Evolução Final")
        ax.legend()
        st.pyplot(fig, use_container_width=False)

        top_individuals = tools.selBest(population, k=20)
        all_best = pd.DataFrame()
        for ind in top_individuals:
            df = tesouros.iloc[ind].copy()
            df["Score"] = ind.fitness.values[0]
            all_best = pd.concat([all_best, df], ignore_index=True)
        all_best.drop_duplicates(inplace=True)
        all_best.sort_values("Score", ascending=False, inplace=True)

        st.markdown("### 🏆 Títulos Otimizados Ordenados por Score")
        st.dataframe(all_best)
        st.download_button("📥 Baixar CSV", all_best.to_csv(index=False).encode("utf-8-sig"), "melhores_titulos.csv")
