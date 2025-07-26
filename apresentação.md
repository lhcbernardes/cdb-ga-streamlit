---
marp: true
---

# Apresentação: Otimizador de Portfólio - Tesouro Direto com Algoritmo Genético

## Introdução

### **Título:** Otimizador de Portfólio - Tesouro Direto com Algoritmo Genético

### **Objetivo:** 
Desenvolver uma aplicação que otimize portfólios de títulos públicos usando algoritmos genéticos multiobjetivo

### **Problema Central:** 
Como encontrar a melhor combinação de títulos do Tesouro Direto considerando retorno, risco e diversificação?

### **Solução:** 
Algoritmo Genético (GA) com operadores inteligentes, controle de diversidade e otimização multiobjetivo (NSGA-II)

---

## Tecnologias Escolhidas

### **Streamlit**
- **Por que escolhemos:** Interface web rápida e intuitiva
- **Vantagem:** Desenvolvimento rápido, visualizações nativas
- **Resultado:** Interface profissional em poucas linhas de código

### **DEAP (Distributed Evolutionary Algorithms in Python)**
- **Por que escolhemos:** Biblioteca especializada em algoritmos evolutivos
- **Vantagem:** Operadores genéticos prontos, alta flexibilidade
- **Resultado:** Implementação robusta e eficiente

### **Pandas + NumPy**
- **Por que escolhemos:** Manipulação eficiente de dados financeiros
- **Vantagem:** Performance otimizada para cálculos matemáticos
- **Resultado:** Processamento rápido de grandes datasets

### **Matplotlib**
- **Por que escolhemos:** Visualizações científicas de qualidade
- **Vantagem:** Gráficos profissionais e customizáveis
- **Resultado:** Análises visuais claras e informativas

---

## Por que Algoritmo Genético Multiobjetivo?

### **Vantagens para Otimização de Portfólios:**

#### **Uso do NSGA-II**
- O algoritmo NSGA-II (Non-dominated Sorting Genetic Algorithm II) foi utilizado por ser referência mundial em otimização multiobjetivo.
- Ele permite otimizar simultaneamente múltiplos critérios conflitantes (retorno, risco, diversidade), sem precisar combinar tudo em um único score.
- O NSGA-II gera uma **fronteira de Pareto**: um conjunto de soluções não-dominadas, oferecendo ao usuário várias opções de portfólios com diferentes trade-offs.
- Benefícios principais:
    - Garante diversidade de soluções
    - Evita convergência para uma única solução dominante
    - Permite ao investidor escolher o portfólio mais adequado ao seu perfil
    - Resultados visualmente interpretáveis (gráfico de Pareto)

#### **1. Problema Combinatório Complexo**
- N títulos → 2^N combinações possíveis
- GA encontra soluções boas sem testar todas as combinações
- **Exemplo:** 50 títulos = 1.125.899.906.842.624 combinações

#### **2. Múltiplos Objetivos Otimizados Simultaneamente**
- O app sempre otimiza **retorno** (max), **risco** (min) e **diversificação** (max) ao mesmo tempo
- Utiliza o algoritmo NSGA-II para gerar uma **fronteira de Pareto**: várias soluções não-dominadas para o usuário escolher
- O usuário pode comparar facilmente os trade-offs entre risco, retorno e diversidade

#### **3. Diversidade Garantida**
- O algoritmo monitora e reforça a diversidade da população
- Evita convergência prematura para soluções únicas
- Garante variedade de portfólios para diferentes perfis de investidor

---

## Visualização e Interatividade

- Interface em abas: Resumo, Fronteira de Pareto, Detalhes, Configurações Avançadas, Ajuda
- Gráfico de Pareto (Risco x Retorno) destacando portfólios não-dominados
- Tabela interativa da fronteira de Pareto com exportação para CSV
- Gráficos de pizza e barras para composição do portfólio
- Feedback visual, tooltips e tutorial integrado

---

## Conclusão

- O app entrega uma solução moderna, robusta e flexível para otimização de portfólios do Tesouro Direto
- O uso do **NSGA-II** garante que múltiplos objetivos sejam otimizados simultaneamente, proporcionando uma fronteira de Pareto rica e opções para diferentes perfis de investidor
- O usuário tem autonomia para escolher o portfólio ideal de acordo com seu perfil de risco e preferência de diversificação
- O uso de algoritmos evolutivos multiobjetivo (NSGA-II) garante resultados superiores e visualmente interpretáveis

---

## Demonstração Prática

### **Cenários de Uso:**

#### **Investidor Conservador:**
- Prefere portfólios com menor risco e maior diversificação
- Pode ajustar os parâmetros avançados para maior elite (ex: ELITE_SIZE: 15%) e menor threshold de diversidade (ex: DIVERSITY_THRESHOLD: 0.2)
- Na Fronteira de Pareto, seleciona portfólios mais à esquerda (menor risco)

#### **Investidor Agressivo:**
- Busca portfólios com maior retorno, mesmo assumindo mais risco
- Pode ajustar os parâmetros avançados para menor elite (ex: ELITE_SIZE: 5%) e maior threshold de diversidade (ex: DIVERSITY_THRESHOLD: 0.5)
- Na Fronteira de Pareto, seleciona portfólios mais à direita (maior retorno)

#### **Investidor Balanceado:**
- Procura equilíbrio entre risco, retorno e diversificação
- Pode usar valores intermediários nos parâmetros avançados
- Na Fronteira de Pareto, escolhe portfólios próximos ao centro

> Todos os perfis podem explorar a Fronteira de Pareto para encontrar o portfólio ideal, de acordo com sua preferência de risco, retorno e diversidade.

### **Resultados Esperados:**
- **Convergência mais rápida** com operadores inteligentes
- **Soluções mais diversas** com controle de diversidade
- **Flexibilidade** para diferentes objetivos
- **Interface intuitiva** para usuários finais

---

## Agradecimentos e Perguntas

---

## Notas para Apresentação:

### **Tempo Estimado: 8-10 minutos**

### **Estrutura Sugerida:**
- **Slides 1-2:** Introdução e tecnologias (2 min)
- **Slides 3-4:** Justificativa e representação (2 min)
- **Slides 5-6:** Operadores e estratégias (2 min)
- **Slides 7-8:** Parâmetros e diversidade (2 min)
- **Slides 9-10:** Resultados e conclusões (2 min)

### **Pontos-chave para enfatizar:**
- Por que GA para este problema específico
- Inovação nos operadores genéticos
- Controle de diversidade como diferencial
- Resultados práticos alcançados
- Flexibilidade para diferentes perfis

### **Demonstração:**
- Mostrar interface em funcionamento
- Explicar parâmetros avançados
- Demonstrar diferentes estratégias
- Apresentar visualizações em tempo real