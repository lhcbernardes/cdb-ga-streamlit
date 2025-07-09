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
    st.code("Banco,Rentabilidade,Prazo,Liquidez\nBanco_A,13.5,365,Diária")

# 📤 Upload
uploaded_file = st.file_uploader("📂 Envie seu arquivo CSV de CDBs", type=["csv"])
if not uploaded_file:
    st.warning("Envie um arquivo CSV com colunas: Banco, Rentabilidade, Prazo, Liquidez")
    st.stop()

# 📊 Carregar e validar dados
cdbs = pd.read_csv(uploaded_file)
expected_cols = {"Banco", "Rentabilidade", "Prazo", "Liquidez"}
if not expected_cols.issubset(set(cdbs.columns)):
    st.error(f"Arquivo CSV inválido. Esperado: {expected_cols}")
    st.stop()

# 🧹 Limpeza e verificação de tipos
cdbs.drop_duplicates(inplace=True)
cdbs.dropna(inplace=True)

try:
    cdbs["Rentabilidade"] = cdbs["Rentabilidade"].astype(float)
    cdbs["Prazo"] = cdbs["Prazo"].astype(int)
except ValueError:
    st.error("Erro: Rentabilidade deve ser numérica e Prazo deve ser inteiro.")
    st.stop()

st.success(f"{len(cdbs)} CDBs carregados com sucesso.")
st.dataframe(cdbs.head())

# ⚙️ Parâmetros GA
st.sidebar.header("⚙️ Parâmetros do Algoritmo Genético")
POP_SIZE = st.sidebar.slider("Tamanho da população", 50, 1000, 200, 50)
NGEN = st.sidebar.slider("Número de gerações", 10, 1000, 300, 10)
CXPB = st.sidebar.slider("Probabilidade de crossover", 0.5, 1.0, 0.7, 0.05)
MUTPB = st.sidebar.slider("Probabilidade de mutação", 0.1, 0.5, 0.3, 0.05)
N_ATIVOS = st.sidebar.slider("CDBs por portfólio", 3, 10, 5)
perfil = st.sidebar.selectbox("Perfil do Investidor", ["Moderado", "Conservador", "Agressivo"])

if len(cdbs) < N_ATIVOS:
    st.error(f"São necessários pelo menos {N_ATIVOS} CDBs no arquivo para formar um portfólio.")
    st.stop()

# 🎯 Função de avaliação com base no perfil
def evaluate(individual):
    selected = cdbs.iloc[individual]
    retorno = selected["Rentabilidade"].mean()
    risco = selected["Rentabilidade"].std()
    prazo = selected["Prazo"].mean()

    if perfil == "Conservador":
        score = (retorno * 0.6) - (risco * 0.4) - (prazo * 0.01)
    elif perfil == "Agressivo":
        score = (retorno * 0.9) - (risco * 0.1)
    else:  # Moderado
        score = (retorno * 0.7) - (risco * 0.3)
    return (score,)

# 🧬 Setup DEAP
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

# 🚀 Função principal
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
        avg_fit = np.mean([ind.fitness.values[0] for ind in population])
        log.append((gen, best.fitness.values[0], avg_fit))

    return population, log

# ▶️ Rodar
if st.button("🚀 Rodar otimização"):
    with st.spinner("Executando..."):
        population, log = run_ga()

        last_gen = log[-1]
        st.success(f"Geração {last_gen[0]}: Score={last_gen[1]:.2f} | Média Score={last_gen[2]:.2f}")

        # 📈 Evolução
        generations, best_scores, avg_scores = zip(*log)
        fig, ax = plt.subplots()
        ax.plot(generations, best_scores, label="Melhor Score")
        ax.plot(generations, avg_scores, label="Média Score", linestyle="--")
        ax.set_xlabel("Geração")
        ax.set_ylabel("Score")
        ax.set_title("Evolução do Algoritmo Genético")
        ax.legend()
        ax.grid(True)
        ax.set_facecolor("#f9f9f9")
        st.pyplot(fig)

        # 📊 Top 5 Portfólios
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
        st.download_button("📥 Baixar CSV", result_df.to_csv(index=False).encode("utf-8-sig"), file_name=filename)
