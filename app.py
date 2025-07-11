
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import random
from datetime import datetime

st.set_page_config(page_title="GA Tesouro Direto Otimizador", layout="wide")
st.title("📈 Otimização dos Melhores Títulos do Tesouro Direto")

# Upload do CSV
with st.expander("📋 Formato esperado do CSV do Tesouro Direto"):
    st.code("Tipo Titulo;Data Vencimento;Data Base;Taxa Compra Manha;...")

uploaded_file = st.file_uploader("📂 Envie o arquivo CSV do Tesouro Direto", type=["csv"])
if not uploaded_file:
    st.info("Aguardando envio do arquivo CSV com colunas como: Tipo Titulo, Taxa Compra Manha, etc.")
    st.stop()

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

# Filtros simplificados
st.sidebar.header("🔍 Filtros de Qualidade")
bancos_excluidos = st.sidebar.multiselect("Excluir títulos de", sorted(tesouros["Banco"].unique()))
rating_opcional = st.sidebar.toggle("Filtrar por Rating", value=True)
ratings_aceitos = []
if rating_opcional:
    ratings_aceitos = st.sidebar.multiselect("Ratings aceitos", sorted(tesouros["Rating"].unique()), default=sorted(tesouros["Rating"].unique()))

# Aplicação dos filtros
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

# Função de avaliação
def evaluate(individual):
    selected = tesouros.iloc[individual]
    retorno = selected["Rentabilidade"].mean()
    return (retorno,)

# Reparar duplicados
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

# DEAP
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
    with st.spinner("Executando..."):
        population = toolbox.population(n=POP_SIZE)
        for ind in population:
            ind.fitness.values = toolbox.evaluate(ind)
        log = []
        for gen in range(1, NGEN + 1):
            offspring = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)
            offspring = [repair_individual(ind) for ind in offspring]
            for ind in offspring:
                ind.fitness.values = toolbox.evaluate(ind)
            population = toolbox.select(population + offspring, k=POP_SIZE)
            best = tools.selBest(population, k=1)[0]
            avg_fit = np.mean([ind.fitness.values[0] for ind in population])
            log.append((gen, best.fitness.values[0], avg_fit))
        last_gen = log[-1]
        st.success(f"Geração {last_gen[0]}: Score={last_gen[1]:.2f} | Média Score={last_gen[2]:.2f}")

        generations, best_scores, avg_scores = zip(*log)
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.plot(generations, best_scores, label="Melhor Score")
        ax.plot(generations, avg_scores, label="Média Score", linestyle="--")
        ax.set_xlabel("Geração")
        ax.set_ylabel("Score")
        ax.set_title("Evolução do Algoritmo")
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
