import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import random

st.title("📈 Otimização de Portfólio de CDBs com Algoritmo Genético (Sem Penalizações)")

# Exibir exemplo de CSV
st.markdown("### 📋 Formato esperado do arquivo CSV:")
st.code("Banco,Rentabilidade,Prazo,Liquidez\nBanco_A,13.5,365,Diária")

# Upload do CSV
uploaded_file = st.file_uploader("📂 Envie seu arquivo CSV de CDBs", type=["csv"])
if not uploaded_file:
    st.warning("Envie um arquivo CSV com colunas: Banco, Rentabilidade, Prazo, Liquidez")
    st.stop()

# Carregar dados
cdbs = pd.read_csv(uploaded_file)
st.success(f"{len(cdbs)} CDBs carregados com sucesso.")
st.dataframe(cdbs.head())

# Parâmetros do GA
st.sidebar.header("⚙️ Parâmetros do Algoritmo Genético")
POP_SIZE = st.sidebar.slider("Tamanho da população", 50, 500, 200, 50)
NGEN = st.sidebar.slider("Número de gerações", 10, 1000, 100, 10)
CXPB = st.sidebar.slider("Probabilidade de crossover", 0.5, 1.0, 0.7, 0.05)
MUTPB = st.sidebar.slider("Probabilidade de mutação", 0.1, 0.5, 0.3, 0.05)

# Setup DEAP
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))  # Max retorno, min risco
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_bool", lambda: 1 if random.random() < 0.05 else 0)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(cdbs))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxUniform, indpb=0.7)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selNSGA2)

# Função de avaliação SEM penalidades
def evaluate(individual):
    selected = cdbs.loc[np.array(individual) == 1]
    retorno = selected['Rentabilidade'].mean() if not selected.empty else 0.0
    risco = selected['Rentabilidade'].std() if len(selected) > 1 else 0.0
    return (retorno, risco)

toolbox.register("evaluate", evaluate)

if st.button("🚀 Rodar otimização"):
    st.info("Executando o algoritmo genético...")

    population = toolbox.population(n=POP_SIZE)
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)

    log = []
    for gen in range(1, NGEN + 1):
        offspring = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)
        for ind in offspring:
            ind.fitness.values = toolbox.evaluate(ind)
        population = toolbox.select(population + offspring, k=POP_SIZE)
        best = tools.sortNondominated(population, k=1, first_front_only=True)[0][0]
        avg_ret = np.mean([ind.fitness.values[0] for ind in population])
        log.append((gen, best.fitness.values[0], best.fitness.values[1], avg_ret))

    # Mostrar apenas a última geração
    last_gen = log[-1]
    st.success(f"Geração {last_gen[0]}: Retorno={last_gen[1]:.2f}, Risco={last_gen[2]:.2f}")

    # Gráfico de evolução
    generations, best_returns, best_risks, avg_returns = zip(*log)
    fig, ax = plt.subplots()
    ax.plot(generations, best_returns, label="Melhor Retorno")
    ax.plot(generations, best_risks, label="Menor Risco")
    ax.plot(generations, avg_returns, label="Média Retorno", linestyle="--")
    ax.set_xlabel("Geração")
    ax.set_ylabel("Valor")
    ax.set_title("Evolução do Algoritmo Genético")
    ax.legend()
    st.pyplot(fig)

    # Exportar os Top 5 portfólios
    top5 = tools.sortNondominated(population, k=5, first_front_only=True)[0]
    result_df = pd.DataFrame()
    for i, ind in enumerate(top5):
        selecionados = cdbs.loc[np.array(ind) == 1].copy()
        selecionados["Portfólio"] = f"Portfólio_{i+1}"
        selecionados["Retorno"] = ind.fitness.values[0]
        selecionados["Risco"] = ind.fitness.values[1]
        result_df = pd.concat([result_df, selecionados], ignore_index=True)

    st.markdown("### 📊 Portfólios Otimizados (Top 5)")
    st.dataframe(result_df)
    st.download_button("📥 Baixar CSV dos Portfólios", result_df.to_csv(index=False).encode("utf-8-sig"), "portfolios_otimizados.csv")
