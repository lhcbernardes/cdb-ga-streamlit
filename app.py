
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import random

st.set_page_config(page_title="GA CDB Otimizador", layout="wide")
st.title("📈 Otimização de Portfólio de CDBs com Algoritmo Genético")

with st.expander("📋 Formato do CSV esperado"):
    st.code("Banco,Rentabilidade,Prazo,Liquidez,Rating\nBanco_A,13.5,365,Diária,A+")

uploaded_file = st.file_uploader("📂 Envie seu arquivo CSV de CDBs", type=["csv"])
if not uploaded_file:
    st.warning("Envie um arquivo CSV com colunas: Banco, Rentabilidade, Prazo, Liquidez, Rating")
    st.stop()

cdbs = pd.read_csv(uploaded_file)
expected_cols = {"Banco", "Rentabilidade", "Prazo", "Liquidez", "Rating"}
if not expected_cols.issubset(set(cdbs.columns)):
    st.error(f"Arquivo CSV inválido. Esperado: {expected_cols}")
    st.stop()

cdbs.drop_duplicates(inplace=True)
cdbs.dropna(inplace=True)
st.success(f"{len(cdbs)} CDBs carregados com sucesso.")
st.dataframe(cdbs.head())

st.sidebar.header("⚙️ Parâmetros do Algoritmo Genético")
POP_SIZE = st.sidebar.slider("Tamanho da população", 50, 1000, 200, 50)
NGEN = st.sidebar.slider("Número de gerações", 10, 1000, 300, 10)
CXPB = st.sidebar.slider("Probabilidade de crossover", 0.5, 1.0, 0.7, 0.05)
MUTPB = st.sidebar.slider("Probabilidade de mutação", 0.1, 0.5, 0.3, 0.05)
N_ATIVOS = st.sidebar.slider("CDBs por portfólio", 3, 10, 5)
perfil = st.sidebar.selectbox("Perfil do Investidor", ["Moderado", "Conservador", "Agressivo"])

st.sidebar.header("🧪 Filtros de Qualidade")
min_rent = st.sidebar.slider("Rentabilidade mínima (% do CDI)", 80.0, 200.0, 100.0, 0.5)
cdi_taxa = st.sidebar.number_input("📈 CDI atual (% a.a.)", min_value=5.0, max_value=20.0, value=10.00, step=0.1)
max_prazo = st.sidebar.slider("Prazo máximo (dias)", 30, 2000, 1000, 30)
liq_opcao = st.sidebar.selectbox("Liquidez desejada", ["Qualquer", "Diária", "No vencimento"])
bancos_excluidos = st.sidebar.multiselect("Excluir bancos emissores", sorted(cdbs["Banco"].unique()))

rating_opcional = st.sidebar.toggle("Filtrar por Rating", value=True)
ratings_aceitos = []
if rating_opcional:
    ratings_aceitos = st.sidebar.multiselect("Ratings aceitos", sorted(cdbs["Rating"].unique()), default=sorted(cdbs["Rating"].unique()))

# Debug dos filtros
st.write("🔍 Total antes dos filtros:", len(cdbs))
st.write("🧾 Rentabilidade mínima ativa:", min_rent)
st.write("📉 CDI atual informado:", cdi_taxa)
st.write("📅 Prazo máximo ativo:", max_prazo)

# Aplicar filtros
rentabilidade_esperada = (min_rent / 100.0) * cdi_taxa
filtro = pd.Series([True] * len(cdbs))
filtro_rent = cdbs["Rentabilidade"] >= rentabilidade_esperada
st.write("✅ CDBs com rentabilidade >= min:", filtro_rent.sum())
filtro_prazo = cdbs["Prazo"] <= max_prazo
st.write("✅ CDBs com prazo <= max:", filtro_prazo.sum())

if liq_opcao != "Qualquer":
    filtro_liq = cdbs["Liquidez"] == liq_opcao
    st.write("✅ CDBs com liquidez:", filtro_liq.sum())
    filtro &= filtro_liq

if bancos_excluidos:
    filtro_bancos = ~cdbs["Banco"].isin(bancos_excluidos)
    st.write("✅ CDBs após excluir bancos:", filtro_bancos.sum())
    filtro &= filtro_bancos

if rating_opcional and ratings_aceitos:
    filtro_rating = cdbs["Rating"].isin(ratings_aceitos)
    st.write("✅ CDBs com rating aceito:", filtro_rating.sum())
    filtro &= filtro_rating

filtro &= filtro_rent & filtro_prazo
cdbs = cdbs[filtro]

if cdbs.empty:
    st.error("❌ Nenhum CDB atende aos critérios selecionados.")
    st.stop()

# Função de avaliação
def evaluate(individual):
    selected = cdbs.iloc[individual]
    retorno = selected["Rentabilidade"].mean()
    risco = selected["Rentabilidade"].std()
    prazo = selected["Prazo"].mean()
    if perfil == "Conservador":
        score = (retorno * 0.6) - (risco * 0.4) - (prazo * 0.01)
    elif perfil == "Agressivo":
        score = (retorno * 0.9) - (risco * 0.1)
    else:
        score = (retorno * 0.7) - (risco * 0.3)
    return (score,)

# DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(len(cdbs)), N_ATIVOS)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.3)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

if st.button("🚀 Rodar otimização"):
    with st.spinner("Executando..."):
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

        last_gen = log[-1]
        st.success(f"Geração {last_gen[0]}: Score={last_gen[1]:.2f} | Média Score={last_gen[2]:.2f}")

        generations, best_scores, avg_scores = zip(*log)
        fig, ax = plt.subplots(figsize=(5, 2.5))
        ax.plot(generations, best_scores, label="Melhor Score")
        ax.plot(generations, avg_scores, label="Média Score", linestyle="--")
        ax.set_xlabel("Geração")
        ax.set_ylabel("Score")
        ax.set_title("Evolução do Algoritmo Genético")
        ax.legend()
        st.pyplot(fig, use_container_width=False)

        top5 = tools.selBest(population, k=5)
        result_df = pd.DataFrame()
        for i, ind in enumerate(top5):
            df = cdbs.iloc[ind].copy()
            df["Portfólio"] = f"Portfólio_{i+1}"
            df["Score"] = ind.fitness.values[0]
            result_df = pd.concat([result_df, df], ignore_index=True)

        st.markdown("### 🏆 Top 5 Portfólios Otimizados")
        st.dataframe(result_df)
        st.download_button("📥 Baixar CSV", result_df.to_csv(index=False).encode("utf-8-sig"), "portfolios_top5.csv")
