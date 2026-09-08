# Predição de Recompra em E-Commerce Brasileiro com Machine Learning e Análise de Sentimento

**TCC — MBA em Data Science e Analytics — USP/Esalq (2026)**

**Autor:** Paulo Pedro Filho
**Orientador:** Prof. João Vitor Matos Gonçalves

---

## Sobre o projeto

Este repositório contém o código desenvolvido para o Trabalho de Conclusão de Curso do MBA em Data Science e Analytics da USP/Esalq.

O trabalho investiga a possibilidade de prever o comportamento de recompra de clientes em um marketplace brasileiro de e-commerce, utilizando exclusivamente dados da primeira transação de cada consumidor e comparando o desempenho de quatro algoritmos de classificação binária.

**Resultado principal:** o modelo XGBoost otimizado alcançou AUC-ROC de 0,804 e AUC-PR de 0,588, com a experiência logística da primeira compra (cumprimento do prazo de entrega) identificada como principal determinante do comportamento de recompra.

---

## Base de dados

**Brazilian E-Commerce Public Dataset by Olist**
Disponível em: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

~100 mil pedidos realizados entre 2016 e 2018 por consumidores brasileiros.
Faça o download e coloque os arquivos CSV na mesma pasta dos scripts antes de executar.

---

## Estrutura do repositório

```
tcc_recompra_olist/
│
├── tcc_completo_comentado.py       # Pipeline principal — execução completa do projeto
├── sentiment_analysis_colab.py     # Análise de sentimento via BERTimbau (Google Colab + GPU)
├── teste_variaveis_candidatas.py   # Teste de poder preditivo de variáveis descartadas
├── teste_valor_primeira_compra.py  # Análise da relação valor da compra x recompra
└── README.md                       # Este arquivo
```

---

## Como executar

### Pré-requisitos

```bash
pip install pandas scikit-learn xgboost lightgbm optuna shap matplotlib
```

### Execução principal

1. Baixe os arquivos CSV da Olist no Kaggle
2. Execute o script de análise de sentimento no Google Colab (`sentiment_analysis_colab.py`) para gerar o arquivo `sentiment_features.csv`
3. Coloque todos os CSVs e o `sentiment_features.csv` na mesma pasta que os scripts
4. Execute o pipeline principal:

```bash
python tcc_completo_comentado.py
```

### Arquivos gerados

| Arquivo | Descrição |
|---|---|
| `df_base_features.csv` | Base de dados final com todas as features |
| `df_elegivel.csv` | Pedidos dos clientes elegíveis |
| `items_cache.csv` | Cache da tabela de itens |
| `graficos_tcc_v2.png` | Curva ROC, Curva PR e importância SHAP |

---

## Resultados

| Modelo | AUC-ROC | AUC-PR |
|---|---|---|
| Regressão Logística | 0,603 | 0,072 |
| Random Forest | 0,768 | 0,357 |
| XGBoost | 0,790 | 0,575 |
| **XGBoost (otimizado)** | **0,804** | **0,588** |
| LightGBM | 0,796 | 0,567 |
| LightGBM (otimizado) | 0,795 | 0,586 |

---

## Features utilizadas

| Feature | Dimensão | Descrição |
|---|---|---|
| `valor_primeira_compra` | Transacional | Valor pago na primeira compra (R$) |
| `recencia_dias` | Temporal | Dias entre a primeira compra e o fim da base |
| `atraso_medio` | Logística | Diferença entre prazo prometido e entrega real (dias) |
| `flag_atraso` | Logística | 1 se houve atraso na entrega, 0 caso contrário |
| `customer_state` | Geográfica | Estado do cliente na primeira compra |
| `review_score_primeira` | Textual | Nota da avaliação da primeira compra (1-5) |
| `sentimento_num` | Textual | Sentimento da avaliação (1=positivo, 0=neutro, -1=negativo) |
| `produto_duravel` | Produto | 1=durável (ciclo longo), 0=não-durável (ciclo curto) |

---

## Observações

- O script `sentiment_analysis_colab.py` deve ser executado no **Google Colab com GPU ativa** (Runtime → Alterar tipo de ambiente de execução → GPU T4)
- O pipeline principal (`tcc_completo_comentado.py`) leva entre 15 e 25 minutos para concluir, sendo a maior parte do tempo consumida pela otimização de hiperparâmetros (Optuna, 50 trials por modelo)
- Os scripts de teste (`teste_variaveis_candidatas.py` e `teste_valor_primeira_compra.py`) requerem que o pipeline principal tenha sido executado previamente, ou que os arquivos `df_elegivel.csv`, `items_cache.csv` e `df_base_features.csv` estejam disponíveis na pasta de trabalho
