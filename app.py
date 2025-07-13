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
import concurrent.futures

st.set_page_config(page_title="GA Tesouro Direto Otimizador", layout="wide")
st.title("\U0001F4C8 Otimização dos Melhores Títulos do Tesouro Direto")

@st.cache_data(show_spinner=False)
def baixar_arquivo_tesouro():
    url = "https://www.tesourotransparente.gov.br/ckan/dataset/df56aa42-484a-4a59-8184-7676580c81e3/resource/796d2059-14e9-44e3-80c9-2d9e30b405c1/download/precotaxatesourodireto.csv"
    tamanho_esperado = 13_119_488
    resposta = requests.get(url, stream=True)
    resposta.raise_for_status()
    conteudo = b""
    baixado = 0
    barra = st.progress(0)
    status = st.empty()

    for bloco in resposta.iter_content(1024):
        if bloco:
            conteudo += bloco
            baixado += len(bloco)
            progresso = min(baixado / tamanho_esperado, 1.0)
            barra.progress(progresso)
            status.text(f"{baixado / 1024 / 1024:.2f} MB / {tamanho_esperado / 1024 / 1024:.2f} MB")

    return conteudo

conteudo = baixar_arquivo_tesouro()
arquivo = BytesIO(conteudo)

try:
    raw_df = pd.read_csv(arquivo, sep=";", encoding="utf-8")
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

st.success(f"{len(tesouros)} títulos com vencimento futuro carregados.")
st.dataframe(tesouros.head())

st.sidebar.header("Parâmetros do Algoritmo Genético")
POP_SIZE = st.sidebar.slider("Tamanho da população", 50, 1000, 200, 50)
NGEN = st.sidebar.slider("Número de gerações", 10, 1000, 300, 10)
CXPB = st.sidebar.slider("Probabilidade de crossover", 0.5, 1.0, 0.7, 0.05)
MUTPB = st.sidebar.slider("Probabilidade de mutação", 0.1, 0.8, 0.4, 0.05)
N_ATIVOS = st.sidebar.slider("Títulos por portfólio", 3, 10, 5)

st.sidebar.header("Filtros")
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
if len(tesouros) < N_ATIVOS:
    st.error("Poucos títulos para otimização. Reduza os filtros ou N_ATIVOS.")
    st.stop()

def avaliar(ind):
    retorno = tesouros.iloc[ind]["Rentabilidade"].mean()
    return (retorno,)

def reparar(ind):
    unico = list(dict.fromkeys(ind))
    faltam = N_ATIVOS - len(unico)
    disponiveis = list(set(range(len(tesouros))) - set(unico))
    unico.extend(random.sample(disponiveis, faltam))
    return creator.Individual(unico)

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

def avaliar_em_paralelo(pop):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        resultados = list(executor.map(aivar_ind_fitness, pop))
    return resultados

def aivar_ind_fitness(ind):
    ind.fitness.values = avaliar(ind)
    return ind

if st.button("\U0001F680 Rodar Otimização"):
    inicio = time.time()
    pop = toolbox.population(n=POP_SIZE)
    pop = avaliar_em_paralelo(pop)

    st.markdown("### \U0001F4C9 Evolução do Score")
    dados_chart = pd.DataFrame(columns=["Melhor", "Média"])
    chart = st.line_chart(dados_chart)

    log = []
    melhor_score = -np.inf
    sem_melhora = 0
    parar_apos = 50

    for gen in range(1, NGEN + 1):
        filhos = algorithms.varAnd(pop, toolbox, cxpb=CXPB, mutpb=MUTPB)
        filhos = [reparar(f) for f in filhos]
        filhos = avaliar_em_paralelo(filhos)
        pop = toolbox.select(pop + filhos, k=POP_SIZE)

        best = tools.selBest(pop, 1)[0]
        media = np.mean([ind.fitness.values[0] for ind in pop])
        log.append((gen, best.fitness.values[0], media))

        if gen % 5 == 0 or gen == 1:
            chart.add_rows(pd.DataFrame([[best.fitness.values[0], media]], columns=["Melhor", "Média"]))

        if best.fitness.values[0] > melhor_score:
            melhor_score = best.fitness.values[0]
            sem_melhora = 0
        else:
            sem_melhora += 1
            if sem_melhora >= parar_apos:
                st.warning(f"Parado na geração {gen}, sem melhoria por {parar_apos} gerações.")
                break

    fim = time.time()
    st.success(f"Finalizado em {fim - inicio:.2f} segundos.")

    melhores = tools.selBest(pop, 10)
    resultado = pd.concat([tesouros.iloc[ind].assign(Score=ind.fitness.values[0]) for ind in melhores])
    st.markdown("### \U0001F3C6 Títulos Otimizados")
    st.dataframe(resultado)
    st.download_button("\U0001F4E5 Baixar CSV", resultado.to_csv(index=False).encode("utf-8-sig"), "melhores_titulos.csv")
