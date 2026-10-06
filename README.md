# Predição de recompra em e-commerce brasileiro com machine learning e análise de sentimento

Código do Trabalho de Conclusão de Curso do MBA em Data Science & Analytics (USP/Esalq).

- Autor: Paulo Pedro Filho
- Orientador: Prof. João Vitor Matos Gonçalves
- Versão entregue: tag `v1.0-entrega`

O trabalho prevê, a partir da primeira compra de cada cliente, se haverá uma nova compra em até 180 dias.
A base é o Brazilian E-Commerce Public Dataset by Olist (Kaggle). Foram comparados regressão logística,
Random Forest, XGBoost e LightGBM, com otimização de hiperparâmetros (Optuna), intervalos de confiança
por bootstrap, ablação de variáveis e interpretação por SHAP. O sentimento das avaliações foi estimado
com o BERTimbau.

## Arquivos

| Arquivo | Função |
|---|---|
| `bertimbau_colab.py` | Estima o sentimento das avaliações com BERTimbau (Google Colab, GPU). Gera `bertimbau_sentimento_reviews.csv`. |
| `bertimbau_sentimento_reviews.csv` | Saída do passo anterior, lida pelo script principal. |
| `tcc_completo_v3.py` | Pipeline principal: base, variáveis, modelos, Optuna, métricas, ablação, SHAP e figuras. |
| `tcc_complemento_regressao.py` | Regressão logística inferencial da especificação final, a que consta no TCC. |
| `requirements.txt` | Bibliotecas e versões utilizadas. |
| `tabela_metricas_v3.csv` | Desempenho dos modelos no conjunto de teste, com intervalos de confiança. |
| `tabela_hiperparametros_v3.csv` | Hiperparâmetros escolhidos pelo Optuna. |
| `tabela_ablacao_v3.csv` | Contribuição de cada bloco de variáveis. |
| `shap_importancia_v3.csv` | Importância das variáveis por SHAP. |
| `tabela_logit_corrigida_v3.csv` | Coeficientes e razões de chances da regressão logística final. |
| `linha_regressao_logistica_v3.csv` | Resumo da regressão logística final. |

## Como reproduzir

1. Baixe o dataset em https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce e coloque os CSVs
   na mesma pasta dos scripts. Os dados brutos do Olist não estão neste repositório.
2. Instale as bibliotecas: `pip install -r requirements.txt`
3. (Opcional) Rode `bertimbau_colab.py` no Google Colab com GPU (T4) para gerar o sentimento
   (cerca de 7 minutos). O resultado já está no repositório (`bertimbau_sentimento_reviews.csv`).
4. Rode `tcc_completo_v3.py`. Gera as tabelas, o SHAP e as quatro figuras. A execução completa leva de 40 a 60 minutos.
5. Rode `tcc_complemento_regressao.py`. Gera a regressão logística do TCC
   (`tabela_logit_corrigida_v3.csv` e `linha_regressao_logistica_v3.csv`).

A regressão reportada no TCC vem do passo 5. O passo 4 também grava `tabela_logit_v3.csv`, com a
especificação inicial, que não é a reportada.

## Ambiente

Testado com Python 3.13 (Anaconda, macOS). Versões das bibliotecas em `requirements.txt`.

## Observações

- As sementes aleatórias são fixas. Diferenças na quarta casa decimal podem ocorrer entre versões
  de bibliotecas (principalmente LightGBM) e em GPU.
- O arquivo `base_modelagem_v3.csv`, com identificadores de clientes, não é publicado.
- Fonte dos dados: OLIST. Brazilian E-Commerce Public Dataset by Olist. Compilado por A. Sionek. Kaggle, 2018.
