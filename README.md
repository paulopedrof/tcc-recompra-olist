# Predição de Recompra em E-Commerce Brasileiro com Machine Learning e Análise de Sentimento

**TCC — MBA em Data Science e Analytics — USP/Esalq (2026)**

**Autor:** Paulo Pedro Filho
**Orientador:** Prof. João Vitor Matos Gonçalves

---

## ⚠️ Dados necessários — baixar antes de executar

**Os arquivos de dados NÃO estão incluídos neste repositório.**

Antes de executar qualquer script, você precisa:

**1. Baixar a base Olist do Kaggle:**
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

Descompacte o ZIP e coloque os 7 arquivos CSV na mesma pasta dos scripts:
- `olist_orders_dataset.csv`
- `olist_customers_dataset.csv`
- `olist_order_payments_dataset.csv`
- `olist_order_reviews_dataset.csv`
- `olist_order_items_dataset.csv`
- `olist_products_dataset.csv`
- `olist_product_category_name_translation.csv`

**2. Gerar o arquivo de sentimento no Google Colab:**

Execute o script `sentiment_analysis_colab.py` no Google Colab com GPU ativa (Runtime → Alterar tipo de ambiente de execução → GPU T4). Ao final, baixe o arquivo `sentiment_features.csv` gerado e coloque-o na mesma pasta dos scripts.

Só então execute o pipeline principal.

---

## Sobre o projeto

Este repositório contém o código desenvolvido para o Trabalho de Conclusão de Curso do MBA em Data Science e Analytics da USP/Esalq.

O trabalho investiga a possibilidade de prever o comportamento de recompra de clientes em um marketplace brasileiro de e-commerce, utilizando exclusivamente dados da primeira transação de cada consumidor e comparando o desempenho de quatro algoritmos de classificação binária.

**Resultado principal:** o modelo XGBoost otimizado alcançou AUC-ROC de 0,804 e AUC-PR de 0,588, com a experiência logística da primeira compra (cumprimento do prazo de entrega) identificada como principal determinante do comportamento de recompra.

---

## Estrutura do repositório

```
tcc_recompra_olist/
│
├── tcc_completo_comentado.py        # Pipeline principal — execução completa
├── sentiment_analysis_colab.py      # Análise de sentimento via BERTimbau (Colab + GPU)
├── teste_variaveis_candidatas.py    # Teste de variáveis descartadas (frete, postagem)
├── teste_valor_primeira_compra.py   # Análise da relação valor da compra x recompra
├── .gitignore                       # Arquivos excluídos do repositório
└── README.md                        # Este arquivo
```

---

## Como executar

### Pré-requisitos

```bash
pip install pandas scikit-learn xgboost lightgbm optuna shap matplotlib
```

### Passo a passo

1. Baixe os CSVs da Olist no Kaggle (link acima)
2. Execute `sentiment_analysis_colab.py` no Google Colab para gerar `sentiment_features.csv`
3. Coloque todos os CSVs e o `sentiment_features.csv` na mesma pasta dos scripts
4. Execute o pipeline principal:

```bash
python tcc_completo_comentado.py
```

O script leva entre 15 e 25 minutos — a maior parte do tempo é a otimização de hiperparâmetros (Optuna, 50 trials por modelo).

### Arquivos gerados automaticamente

| Arquivo | Descrição |
|---|---|
| `df_base_features.csv` | Base de dados final com todas as features |
| `df_elegivel.csv` | Pedidos dos clientes elegíveis (uso nos scripts de teste) |
| `items_cache.csv` | Cache da tabela de itens (uso nos scripts de teste) |
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
| `review_score_primeira` | Textual | Nota da avaliação da primeira compra (1 a 5) |
| `sentimento_num` | Textual | Sentimento da avaliação (1=positivo, 0=neutro, -1=negativo) |
| `produto_duravel` | Produto | 1=durável (ciclo longo), 0=não-durável (ciclo curto) |

---

## Observações

- O script `sentiment_analysis_colab.py` **deve ser executado no Google Colab com GPU ativa** — não no ambiente local
- Os scripts de teste requerem que o pipeline principal tenha rodado primeiro, ou que os arquivos `df_elegivel.csv`, `items_cache.csv` e `df_base_features.csv` estejam disponíveis na pasta
- Os CSVs da Olist e os arquivos gerados estão no `.gitignore` e não são versionados
