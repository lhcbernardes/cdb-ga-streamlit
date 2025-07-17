# Otimizador de Portfólio - Tesouro Direto com Algoritmo Genético

## 📊 Descrição

Aplicação Streamlit para otimização de portfólios de Tesouro Direto usando Algoritmo Genético (GA) avançado. O sistema utiliza dados oficiais do Tesouro Nacional para encontrar as melhores combinações de títulos públicos.

## 🚀 Melhorias Implementadas no Algoritmo Genético

### 1. **Operadores Genéticos Melhorados**

#### Crossover Uniforme Inteligente
- **Implementação**: `crossover_uniforme()`
- **Melhoria**: Usa máscara aleatória para decidir de qual pai herdar cada posição
- **Benefício**: Preserva diversidade e evita convergência prematura

#### Mutação Inteligente
- **Implementação**: `mutacao_inteligente()`
- **Melhoria**: Substitui apenas 30% dos títulos, preservando títulos bons
- **Benefício**: Mantém características promissoras enquanto explora novas combinações

#### Mutação por Troca
- **Implementação**: `mutacao_swap()`
- **Melhoria**: Troca posições de títulos existentes
- **Benefício**: Explora diferentes ordenações sem perder diversidade

### 2. **Controle de Diversidade**

#### Cálculo de Diversidade
- **Implementação**: `calcular_diversidade()`
- **Método**: Baseado na sobreposição de títulos entre indivíduos
- **Benefício**: Monitora convergência da população

#### Reinicialização Adaptativa
- **Trigger**: Quando diversidade < threshold
- **Ação**: Reinicializa 20% da população
- **Benefício**: Evita convergência prematura

### 3. **Estratégias de Avaliação Expandidas**

#### Sharpe Ratio
- **Fórmula**: `retorno / risco`
- **Benefício**: Otimiza retorno ajustado pelo risco

#### Multi-Objetivo
- **Fórmula**: `retorno - 0.5 * risco + 0.3 * diversidade`
- **Benefício**: Balanceia múltiplos objetivos simultaneamente

### 4. **Parâmetros Avançados**

#### Controle de Elite
- **Parâmetro**: `ELITE_SIZE` (5-20%)
- **Benefício**: Preserva os melhores indivíduos

#### Tamanho do Torneio
- **Parâmetro**: `TOURNAMENT_SIZE` (2-8)
- **Benefício**: Controla pressão seletiva

#### Limiar de Diversidade
- **Parâmetro**: `DIVERSITY_THRESHOLD` (0.1-0.9)
- **Benefício**: Controla quando reinicializar população

### 5. **Early Stopping Inteligente**

#### Critério de Parada
- **Condição**: Sem melhoria por 20 gerações
- **Benefício**: Evita computação desnecessária

#### Monitoramento em Tempo Real
- **Métricas**: Melhor score, média da população
- **Visualização**: Gráfico de evolução atualizado

### 6. **Análise de Resultados**

#### Gráfico de Pareto
- **Eixo X**: Risco (Desvio Padrão)
- **Eixo Y**: Retorno Médio
- **Benefício**: Visualiza trade-off risco-retorno

#### Análise de Diversidade
- **Métrica**: Diversidade da população final
- **Histograma**: Distribuição de scores
- **Benefício**: Avalia qualidade da convergência

## 🔧 Instalação e Uso

### Dependências
```bash
pip install streamlit pandas numpy matplotlib deap requests
```

### Execução
```bash
streamlit run app.py
```

## 📈 Estratégias de Score Disponíveis

1. **Média da Rentabilidade**: Média simples das taxas
2. **Rentabilidade Total até o Vencimento**: Retorno composto
3. **Rentabilidade Ajustada pelo Prazo**: Penaliza prazos longos
4. **Diversificação de Tipos**: Bônus por variedade
5. **Sharpe Ratio**: Retorno ajustado pelo risco
6. **Multi-Objetivo**: Balanceia retorno, risco e diversificação

## 🎯 Benefícios das Melhorias

### Performance
- **Convergência mais rápida** com operadores inteligentes
- **Menos computação** com early stopping
- **Melhor qualidade** de soluções encontradas

### Robustez
- **Diversidade mantida** com reinicialização adaptativa
- **Convergência controlada** com parâmetros avançados
- **Múltiplas estratégias** para diferentes objetivos

### Usabilidade
- **Interface intuitiva** com controles avançados
- **Visualizações em tempo real** da evolução
- **Análise detalhada** dos resultados

## 🔬 Aspectos Técnicos

### Representação do Indivíduo
- **Estrutura**: Lista de índices únicos
- **Tamanho**: N_ATIVOS títulos por portfólio
- **Reparo**: Remove duplicatas automaticamente

### Operadores Genéticos
- **Seleção**: Torneio com tamanho configurável
- **Crossover**: Uniforme com máscara aleatória
- **Mutação**: Inteligente + Swap
- **Elitismo**: Preserva percentual configurável

### Função de Avaliação
- **Múltiplas estratégias** implementadas
- **Tratamento de erros** robusto
- **Normalização** adequada dos scores

## 📊 Métricas de Qualidade

### Diversidade Populacional
- **Cálculo**: Distância média entre indivíduos
- **Monitoramento**: Contínuo durante evolução
- **Ação**: Reinicialização quando necessário

### Early Stopping
- **Critério**: Gerações sem melhoria
- **Limite**: Configurável (padrão: 20)
- **Benefício**: Economia computacional

### Análise de Resultados
- **Gráfico de Pareto**: Risco vs. Retorno
- **Distribuição de Scores**: Histograma da população final
- **Métricas Detalhadas**: Estatísticas do melhor portfólio

## 🚀 Próximas Melhorias Sugeridas

1. **Algoritmos Multi-Objetivo**: NSGA-II, SPEA2
2. **Machine Learning**: Predição de performance
3. **Otimização Multi-Período**: Considerar diferentes cenários
4. **Integração com APIs**: Dados em tempo real
5. **Backtesting**: Validação histórica de estratégias

## 📝 Licença

Este projeto é de código aberto e está disponível sob a licença MIT.
