---
marp: true
---

# Apresentação: Otimizador de Portfólio - Tesouro Direto com Algoritmo Genético

## **Slide 1: Introdução**

### **Título:** Otimizador de Portfólio - Tesouro Direto com Algoritmo Genético

### **Objetivo:** 
Desenvolver uma aplicação que otimize portfólios de títulos públicos usando algoritmos genéticos

### **Problema Central:** 
Como encontrar a melhor combinação de títulos do Tesouro Direto considerando retorno, risco e diversificação?

### **Solução:** 
Algoritmo Genético (GA) com operadores inteligentes e controle de diversidade

---

## 🛠️ **Slide 2: Tecnologias Escolhidas**

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

## �� **Slide 3: Por que Algoritmo Genético?**

### **Vantagens para Otimização de Portfólios:**

#### **1. Problema Combinatório Complexo**
- N títulos → 2^N combinações possíveis
- GA encontra soluções boas sem testar todas as combinações
- **Exemplo:** 50 títulos = 1.125.899.906.842.624 combinações

#### **2. Múltiplos Objetivos**
- Retorno vs. Risco vs. Diversificação
- GA naturalmente lida com trade-offs
- Balanceia objetivos conflitantes

#### **3. Flexibilidade**
- Fácil adicionar novas restrições
- Adaptável a diferentes estratégias de investimento
- Múltiplas funções de fitness

#### **4. Convergência Global**
- Evita ótimos locais
- Explora espaço de soluções eficientemente
- Mantém diversidade populacional

---

## 📊 **Slide 4: Representação do Problema**

### **Indivíduo (Portfólio)**
```python
# Representação: Lista de índices únicos
individuo = [0, 15, 23, 7, 42]  # 5 títulos selecionados
```

### **População**
```python
# Múltiplos portfólios candidatos
populacao = [
    [0, 15, 23, 7, 42],   # Portfólio 1
    [5, 12, 30, 8, 19],   # Portfólio 2
    [2, 18, 25, 11, 33],  # Portfólio 3
    # ... mais portfólios
]
```

### **Fitness (Score)**
```python
# Múltiplas estratégias de avaliação
estrategias = [
    "Média da Rentabilidade",
    "Sharpe Ratio", 
    "Multi-Objetivo"
]
```

### **Dados Reais**
- Fonte: Tesouro Nacional (API oficial)
- Atualização: Dados em tempo real
- Filtros: Apenas títulos com vencimento futuro

---

## 🔄 **Slide 5: Operadores Genéticos Implementados**

### **1. Crossover Uniforme Inteligente**
```python
def crossover_uniforme(ind1, ind2):
    # Máscara aleatória decide herança
    mask = [random.random() < 0.5 for _ in range(N_ATIVOS)]
    # Preserva diversidade, evita convergência prematura
```

**Inovação:** Usa máscara aleatória para decidir de qual pai herdar cada posição

### **2. Mutação Inteligente**
```python
def mutacao_inteligente(ind, indpb=0.3):
    # Substitui apenas 30% dos títulos
    # Preserva títulos bons, explora novas combinações
```

**Inovação:** Preserva 70% dos títulos bons, substitui apenas 30%

### **3. Mutação por Troca**
```python
def mutacao_swap(ind, indpb=0.2):
    # Troca posições de títulos existentes
    # Explora diferentes ordenações
```

**Inovação:** Mantém os mesmos títulos, mas em ordem diferente

---

## 🎯 **Slide 6: Estratégias de Avaliação**

### **Estratégias Implementadas:**

#### **1. Média da Rentabilidade**
- Simples e direta
- Ideal para iniciantes
- Fórmula: `mean(rentabilidades)`

#### **2. Sharpe Ratio**
- Retorno ajustado pelo risco
- Métrica profissional
- Fórmula: `retorno / risco`

#### **3. Multi-Objetivo**
- Balanceia retorno, risco e diversificação
- Abordagem holística
- Fórmula: `retorno - 0.5 * risco + 0.3 * diversidade`

### **Por que múltiplas estratégias?**
- Diferentes perfis de investidor
- Diferentes objetivos de investimento
- Flexibilidade na otimização
- Comparação de resultados

---

## ⚙️ **Slide 7: Parâmetros Avançados**

### **Controle de Elite (5-20%)**
```python
ELITE_SIZE = st.sidebar.slider("Tamanho da Elite (%)", 5, 20, 10)
```
- **Baixo (5%)**: Mais exploração, menos convergência
- **Alto (20%)**: Mais convergência, menos diversidade
- **Médio (10%)**: Equilíbrio entre exploração e exploração

### **Tamanho do Torneio (2-8)**
```python
TOURNAMENT_SIZE = st.sidebar.slider("Tamanho do Torneio", 2, 8, 4)
```
- **Baixo (2)**: Baixa pressão seletiva, mais diversidade
- **Alto (8)**: Alta pressão seletiva, convergência rápida

### **Limiar de Diversidade (0.1-0.9)**
```python
DIVERSITY_THRESHOLD = st.sidebar.slider("Limiar de Diversidade", 0.1, 0.9, 0.3)
```
- **Baixo (0.1)**: Raramente reinicializa
- **Alto (0.9)**: Reinicializa frequentemente

---

## 📈 **Slide 8: Controle de Diversidade**

### **Problema Identificado:** Convergência Prematura
- População fica muito similar
- Perde capacidade de exploração
- Fica preso em ótimos locais

### **Solução Implementada:**
```python
def calcular_diversidade(populacao):
    # Calcula distância média entre indivíduos
    # Baseado na sobreposição de títulos
    return np.mean(distancias)

# Reinicialização adaptativa
if diversidade < DIVERSITY_THRESHOLD:
    reinicializar_20%_da_populacao()
```

### **Benefícios Alcançados:**
- ✅ Mantém exploração ativa
- ✅ Evita ótimos locais
- ✅ Melhora qualidade das soluções
- ✅ Convergência mais robusta

---

## 🚀 **Slide 9: Resultados e Visualizações**

### **Gráfico de Evolução**
- Melhor score vs. Geração
- Média da população
- Monitoramento em tempo real
- Early stopping inteligente

### **Gráfico de Pareto**
- Risco vs. Retorno
- Visualiza trade-offs
- Identifica fronteira eficiente
- Destaca melhor portfólio

### **Análise de Diversidade**
- Métricas da população final
- Histograma de scores
- Qualidade da convergência
- Distribuição de resultados

### **Métricas Detalhadas**
- Score final do melhor portfólio
- Rentabilidade média
- Prazo médio
- Diversidade de títulos
- Risco (desvio padrão)

---

## 🎉 **Slide 10: Conclusões e Benefícios**

### **O que foi alcançado:**

#### ✅ **Algoritmo Robusto**
- Operadores genéticos inteligentes
- Controle de diversidade implementado
- Early stopping eficiente
- Múltiplas estratégias de avaliação

#### ✅ **Interface Intuitiva**
- Parâmetros configuráveis
- Visualizações em tempo real
- Múltiplas estratégias
- Análise detalhada de resultados

#### ✅ **Resultados Qualitativos**
- Melhor convergência
- Soluções mais diversas
- Performance otimizada
- Flexibilidade para diferentes perfis

### **Inovações Implementadas:**
1. **Crossover Uniforme Inteligente**
2. **Mutação Inteligente (30% substituição)**
3. **Controle de Diversidade Adaptativo**
4. **Múltiplas Estratégias de Avaliação**
5. **Parâmetros Avançados Configuráveis**

### **Próximos Passos:**
- Algoritmos multi-objetivo (NSGA-II, SPEA2)
- Machine Learning para predição de performance
- Backtesting histórico de estratégias
- Integração com APIs de dados em tempo real

---

## 📊 **Slide 11: Demonstração Prática**

### **Cenários de Uso:**

#### **Investidor Conservador:**
- ELITE_SIZE: 15%
- TOURNAMENT_SIZE: 6
- DIVERSITY_THRESHOLD: 0.2
- Estratégia: Sharpe Ratio

#### **Investidor Agressivo:**
- ELITE_SIZE: 5%
- TOURNAMENT_SIZE: 2
- DIVERSITY_THRESHOLD: 0.5
- Estratégia: Multi-Objetivo

#### **Investidor Equilibrado:**
- ELITE_SIZE: 10%
- TOURNAMENT_SIZE: 4
- DIVERSITY_THRESHOLD: 0.3
- Estratégia: Média da Rentabilidade

### **Resultados Esperados:**
- **Convergência mais rápida** com operadores inteligentes
- **Soluções mais diversas** com controle de diversidade
- **Flexibilidade** para diferentes objetivos
- **Interface intuitiva** para usuários finais

---

## �� **Slide 12: Agradecimentos e Perguntas**

### **Tecnologias Utilizadas:**
- **Streamlit** - Interface web
- **DEAP** - Algoritmos evolutivos
- **Pandas/NumPy** - Manipulação de dados
- **Matplotlib** - Visualizações

### **Conceitos Aplicados:**
- **Algoritmo Genético** - Otimização
- **Controle de Diversidade** - Prevenção de convergência prematura
- **Multi-Objetivo** - Balanceamento de objetivos
- **Early Stopping** - Otimização de performance

### **Contribuições:**
- Operadores genéticos inovadores
- Sistema de controle de diversidade
- Interface configurável
- Múltiplas estratégias de avaliação

### **Perguntas e Discussão**

---

## 📝 **Notas para Apresentação:**

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