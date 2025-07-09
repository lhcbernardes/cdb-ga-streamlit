import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import random
from datetime import datetime

st.set_page_config(page_title="GA CDB Otimizador", layout="wide")
st.title("📈 Otimização de Portfólio de CDBs com Algoritmo Genético")

# 📌 Exemplo CSV esperado
with st.expander("📋 Formato do CSV esperado"):
    st.code("Banco,Rentabilidade,Prazo,Liquidez,Rating\nBanco_A,13.5,365,Diária,A+")

uploaded_file = st.file_uploader("📂 Envie seu arquivo CSV de CDBs", type=["csv"])
if not uploaded_file:
    st.warning("Envie um arquivo CSV com colunas: Banco, Rentabilidade, Prazo, Liquidez")
    st.stop()

cdbs = pd.read_csv(uploaded_file)
expected_cols = {"Banco", "Rentabilidade", "Prazo", "Liquidez"}
if not expected_cols.issubset(set(cdbs.columns)):
    st.error(f"Arquivo CSV inválido. Esperado: {expected_cols}")
    st.stop()

cdbs.drop_duplicates(inplace=True)
cdbs.dropna(subset=list(expected_cols), inplace=True)

try:
    cdbs["Rentabilidade"] = cdbs["Rentabilidade"].astype(float)
    cdbs["Prazo"] = cdbs["Prazo"].astype(int)
except ValueError:
    st.error("Erro: Rentabilidade deve ser numérica e Prazo deve ser inteiro.")
    st.stop()

# 💬 Filtros adicionais
st.sidebar.header("📋 Filtros de Qualidade")
min_rent = st.sidebar.slider("Rentabilidade mínima (% do CDI)", 80.0, 200.0, 100.0, 0.5)
max_prazo = st.sidebar.slider("Prazo máximo (dias)", 30, 2000, 1000, 30)
liq_opcao = st.sidebar.selectbox("Liquidez desejada", ["Qualquer", "Diária", "No vencimento"])

# ❌ Exclusão de emissores
bancos_disponiveis = sorted(cdbs["Banco"].unique())
bancos_excluidos = st.sidebar.multiselect("Excluir bancos emissores", bancos_disponiveis)

# ✅ Filtrar por rating (se existir)
rating_opcional = "Rating" in cdbs.columns
if rating_opcional:
    ratings = sorted(cdbs["Rating"].dropna().unique())
    ratings_aceitos = st.sidebar.multiselect("Ratings aceitos", ratings, default=ratings)
else:
    ratings_aceitos = []

# Aplicar filtros
filtro = (cdbs["Rentabilidade"] >= min_rent) & (cdbs["Prazo"] <= max_prazo)
if liq_opcao != "Qualquer":
    filtro &= (cdbs["Liquidez"].str.lower() == liq_opcao.lower())
if bancos_excluidos:
    filtro &= ~cdbs["Banco"].isin(bancos_excluidos)
if rating_opcional and ratings_aceitos:
    filtro &= cdbs["Rating"].isin(ratings_aceitos)

cdbs = cdbs[filtro]

if len(cdbs) == 0:
    st.error("Nenhum CDB atende aos critérios selecionados.")
    st.stop()

# 🧲 Cálculo de rentabilidade líquida com IR
def calcular_rentabilidade_liquida(rentabilidade, prazo):
    if prazo <= 180:
        aliquota = 0.225
    elif prazo <= 360:
        aliquota = 0.20
    elif prazo <= 720:
        aliquota = 0.175
    else:
        aliquota = 0.15
    return rentabilidade * (1 - aliquota)

cdbs["Rentabilidade Líquida"] = cdbs.apply(lambda row: calcular_rentabilidade_liquida(row["Rentabilidade"], row["Prazo"]), axis=1)

st.success(f"{len(cdbs)} CDBs qualificados carregados.")
st.dataframe(cdbs)

# ⚙️ Parâmetros do GA
st.sidebar.header("⚙️ Parâmetros do Algoritmo Genético")
POP_SIZE = st.sidebar.slider("Tamanho da população", 50, 1000, 200, 50)
NGEN = st.sidebar.slider("Número de gerações", 10, 1000, 300, 10)
CXPB = st.sidebar.slider("Probabilidade de crossover", 0.5, 1.0, 0.7, 0.05)
MUTPB = st.sidebar.slider("Probabilidade de mutação", 0.1, 0.5, 0.3, 0.05)
N_ATIVOS = st.sidebar.slider("CDBs por portfólio", 3, 10, 5)
perfil = st.sidebar.selectbox("Perfil do Investidor", ["Moderado", "Conservador", "Agressivo"])

if len(cdbs) < N_ATIVOS:
    st.error("Número insuficiente de CDBs qualificados para formar um portfólio.")
    st.stop()

# 🎯 Função de avaliação
def evaluate(individual):
    selected = cdbs.iloc[individual]
    retorno = selected["Rentabilidade Líquida"].mean()
    risco = selected["Rentabilidade Líquida"].std()
    prazo = selected["Prazo"].mean()

    if perfil == "Conservador":
        score = (retorno * 0.6) - (risco * 0.4) - (prazo * 0.01)
    elif perfil == "Agressivo":
        score = (retorno * 0.9) - (risco * 0.1)
    else:
        score = (retorno * 0.7) - (risco * 0.3)
    return (score,)

# 🧬 DEAP Setup
if "FitnessMax" not in creator.__dict__:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(len(cdbs)), N_ATIVOS)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.3)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

def run_ga():
    population = toolbox.population(n=POP_SIZE)
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)

    log = []
    for gen in range(1, NGEN + 1):
        offspring = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)
        for ind in offspring:
            ind.fitness.values = toolbox.evaluate(ind)
        population = toolbox.select(population + offspring, k=POP_SIZE)

        best = tools.selBest(population, k=1)[0]
        avg = np.mean([ind.fitness.values[0] for ind in population])
        log.append((gen, best.fitness.values[0], avg))
    return population, log

if st.button("🚀 Rodar otimização"):
    with st.spinner("Executando..."):
        population, log = run_ga()
        best_gen = log[-1]
        st.success(f"Geração {best_gen[0]}: Score={best_gen[1]:.2f} | Média Score={best_gen[2]:.2f}")

        gens, bests, avgs = zip(*log)
        fig, ax = plt.subplots()
        ax.plot(gens, bests, label="Melhor Score")
        ax.plot(gens, avgs, label="Média Score", linestyle="--")
        ax.set_xlabel("Geração")
        ax.set_ylabel("Score")
        ax.set_title("Evolução do Algoritmo Genético")
        ax.legend()
        ax.grid(True)
        ax.set_facecolor("#f9f9f9")
        st.pyplot(fig)

        top5 = tools.selBest(population, k=5)
        result_df = pd.DataFrame()
        for i, ind in enumerate(top5):
            df = cdbs.iloc[ind].copy()
            df["Portfólio"] = f"Portfólio_{i+1}"
            df["Score"] = ind.fitness.values[0]
            result_df = pd.concat([result_df, df], ignore_index=True)

        st.markdown("### 🏆 Top 5 Portfólios Otimizados")
        st.dataframe(result_df)

        filename = f"portfolios_top5_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        st.download_button("📅 Baixar CSV", result_df.to_csv(index=False).encode("utf-8-sig"), file_name=filename)
