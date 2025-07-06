import streamlit as st
import pandas as pd
import numpy as np
import random
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt

st.title("💼 Otimização de Portfólios de CDBs - Algoritmo Genético")

# Exemplo de CSV
with st.expander("📌 Exemplo de CSV esperado"):
    st.code("""Banco,Rentabilidade,Prazo,Liquidez
Banco_1,13.5,365,Diária
Banco_2,15.2,720,Vencimento
Banco_3,12.8,540,Mensal""", language='csv')

uploaded_file = st.file_uploader("🔹 Envie seu arquivo CSV de CDBs", type="csv")
if uploaded_file:
    cdbs = pd.read_csv(uploaded_file)
    st.dataframe(cdbs.head())

    # Parâmetros ajustáveis
    st.sidebar.header("⚙️ Parâmetros do GA")
    POP_SIZE = st.sidebar.slider("Tamanho da população", 50, 500, 200)
    NGEN = st.sidebar.slider("Número de gerações", 10, 300, 100)
    CXPB = st.sidebar.slider("Probabilidade de crossover", 0.0, 1.0, 0.7)
    MUTPB = st.sidebar.slider("Probabilidade de mutação", 0.0, 1.0, 0.3)

    st.sidebar.header("⚙️ Restrições")
    max_por_banco = st.sidebar.slider("Máximo de CDBs por banco", 1, 5, 2)
    max_prazo = st.sidebar.slider("Prazo médio máximo (dias)", 180, 1500, 1000)
    min_liquidez_diaria = st.sidebar.slider("Mínimo de CDBs com liquidez Diária", 1, 5, 1)

    # Função de avaliação
    def evaluate(individual):
        selected = cdbs.loc[np.array(individual) == 1]

        if len(selected) != 5:
            return (0.0, 9999.0)
        if selected['Banco'].value_counts().max() > max_por_banco:
            return (0.0, 9999.0)
        if selected['Prazo'].mean() > max_prazo:
            return (0.0, 9999.0)
        if (selected['Liquidez'] == 'Diária').sum() < min_liquidez_diaria:
            return (0.0, 9999.0)

        retorno = selected['Rentabilidade'].mean()
        risco = selected['Rentabilidade'].std() if len(selected) > 1 else 5.0
        return (retorno, risco)

    # Configuração DEAP
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", lambda: 1 if random.random() < 0.05 else 0)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(cdbs))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxUniform, indpb=0.7)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("evaluate", evaluate)

    if st.button("🚀 Rodar otimização"):
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
            log.append((gen, best.fitness.values[0], best.fitness.values[1]))

            st.write(f"Geração {gen}: Retorno={best.fitness.values[0]:.2f}, Risco={best.fitness.values[1]:.2f}")

        # Gráfico evolução
        df_log = pd.DataFrame(log, columns=['Geração', 'Retorno', 'Risco'])
        st.line_chart(df_log.set_index('Geração'))

        # Gráfico Pareto
        pareto = [ind.fitness.values for ind in population]
        pareto = np.array([p for p in pareto if p[1] < 9999.0])
        fig, ax = plt.subplots()
        ax.scatter(pareto[:,1], pareto[:,0])
        ax.set_xlabel("Risco (Desvio Padrão)")
        ax.set_ylabel("Retorno Médio")
        ax.set_title("Fronteira de Pareto Final")
        st.pyplot(fig)

        # Exportar top portfólios
        top5 = tools.sortNondominated(population, k=5, first_front_only=True)[0]
        result_df = pd.DataFrame()
        for i, ind in enumerate(top5):
            selecionados = cdbs.loc[np.array(ind) == 1].copy()
            selecionados['Portfólio'] = f'Portfolio_{i+1}'
            selecionados['Retorno Médio'] = ind.fitness.values[0]
            selecionados['Risco'] = ind.fitness.values[1]
            result_df = pd.concat([result_df, selecionados], ignore_index=True)

        st.dataframe(result_df)
        csv = result_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("⬇️ Baixar Portfólios Otimizados", csv, "portfolios_otimizados.csv", "text/csv")
