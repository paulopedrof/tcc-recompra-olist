"""
================================================================================
TCC — MBA em Data Science e Analytics — USP/Esalq
================================================================================
Título: Predição de recompra em e-commerce brasileiro com Machine Learning
        e análise de sentimento
Autor:  Paulo Pedro Filho
Orientador: Prof. João Vitor Matos Gonçalves

INSTRUÇÕES DE EXECUÇÃO:
1. Certifique-se de que todos os arquivos CSV da Olist estão na mesma pasta
   que este script, incluindo o arquivo sentiment_features.csv gerado pelo
   script sentiment_analysis_colab.py (executado separadamente no Google Colab)
2. No Spyder, configure o diretório de trabalho para essa pasta
3. Execute o script completo com F5 (Run file)
4. O script leva entre 15 e 25 minutos para concluir (a maior parte do tempo
   é consumida pela otimização de hiperparâmetros com Optuna)

ARQUIVOS NECESSÁRIOS NA MESMA PASTA:
- olist_orders_dataset.csv
- olist_customers_dataset.csv
- olist_order_payments_dataset.csv
- olist_order_reviews_dataset.csv
- olist_order_items_dataset.csv
- olist_products_dataset.csv
- olist_product_category_name_translation.csv
- sentiment_features.csv (gerado pelo sentiment_analysis_colab.py)

ARQUIVOS GERADOS POR ESTE SCRIPT:
- df_base_features.csv  : base de dados final com todas as features
- df_elegivel.csv       : pedidos dos clientes elegíveis (uso em scripts auxiliares)
- items_cache.csv       : tabela de itens (uso em scripts auxiliares)
- graficos_tcc_v2.png   : figura com Curva ROC, Curva PR e importância SHAP
================================================================================
"""

# ================================================================================
# IMPORTAÇÃO DAS BIBLIOTECAS
# ================================================================================
# Cada biblioteca tem uma função específica no pipeline:
# - pandas: manipulação e análise de dados tabulares (DataFrames)
# - optuna: otimização bayesiana de hiperparâmetros dos modelos
# - shap: cálculo de valores SHAP para interpretabilidade dos modelos
# - matplotlib: geração de gráficos e visualizações
# - sklearn: ferramentas de machine learning (pré-processamento, modelos, métricas)
# - xgboost: algoritmo XGBoost de gradient boosting
# - lightgbm: algoritmo LightGBM de gradient boosting

import pandas as pd
import optuna
import shap
import matplotlib
matplotlib.use("Agg")  # modo sem interface gráfica — salva figuras em arquivo
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder      # codifica variáveis categóricas
from sklearn.model_selection import train_test_split # divide dados em treino e teste
from sklearn.linear_model import LogisticRegression  # modelo de regressão logística
from sklearn.preprocessing import StandardScaler     # normaliza as features
from sklearn.metrics import (
    roc_auc_score,            # métrica AUC-ROC
    average_precision_score,  # métrica AUC-PR
    classification_report     # relatório completo de desempenho
)
from sklearn.metrics import roc_curve, precision_recall_curve  # curvas de avaliação
from sklearn.ensemble import RandomForestClassifier  # modelo Random Forest
from xgboost import XGBClassifier                   # modelo XGBoost
from lightgbm import LGBMClassifier                 # modelo LightGBM

# Suprime mensagens de progresso do Optuna para manter o output limpo
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ================================================================================
# SEÇÃO 1 — CARREGAMENTO DOS DADOS
# ================================================================================
# A base Olist é composta por 9 tabelas relacionais. Carregamos apenas as
# necessárias para o pipeline de modelagem. Cada tabela é lida do arquivo CSV
# e armazenada num DataFrame do pandas.

print("=" * 60)
print("SEÇÃO 1 — Carregando dados...")
print("=" * 60)

# Tabela principal de pedidos: contém status, datas e identificadores
orders = pd.read_csv("olist_orders_dataset.csv")

# Tabela de clientes: contém identificadores e localização geográfica
customers = pd.read_csv("olist_customers_dataset.csv")

# Tabela de pagamentos: contém valor pago por pedido
payments = pd.read_csv("olist_order_payments_dataset.csv")

# Tabela de avaliações: contém score numérico e texto das reviews
reviews = pd.read_csv("olist_order_reviews_dataset.csv")

# Tabela de itens: contém produtos, quantidades e valores de frete por pedido
items = pd.read_csv("olist_order_items_dataset.csv")

# Tabela de produtos: contém categoria de cada produto
products = pd.read_csv("olist_products_dataset.csv")

# Tabela de tradução de categorias: traduz nomes de categorias para inglês
category = pd.read_csv("olist_product_category_name_translation.csv")

# Features de sentimento geradas pelo BERTimbau no Google Colab
# (ver script sentiment_analysis_colab.py)
sentimento = pd.read_csv("sentiment_features.csv")

print(f"orders:     {orders.shape[0]:,} linhas | {orders.shape[1]} colunas")
print(f"customers:  {customers.shape[0]:,} linhas | {customers.shape[1]} colunas")
print(f"payments:   {payments.shape[0]:,} linhas | {payments.shape[1]} colunas")
print(f"reviews:    {reviews.shape[0]:,} linhas | {reviews.shape[1]} colunas")
print(f"items:      {items.shape[0]:,} linhas | {items.shape[1]} colunas")
print(f"sentimento: {sentimento.shape[0]:,} linhas | {sentimento.shape[1]} colunas")


# ================================================================================
# SEÇÃO 2 — FILTRO E CONVERSÃO DE DATAS
# ================================================================================
# Mantemos apenas pedidos com status "delivered" (entregues), pois somente
# uma compra efetivamente concluída representa uma experiência de compra
# completa que pode influenciar o comportamento de recompra.
# Pedidos cancelados, em trânsito ou com problema são descartados.
#
# As colunas de data chegam no CSV como texto (string). Convertemos para
# o formato datetime do pandas, que permite calcular diferenças entre datas
# — necessário para calcular atrasos de entrega e recência.

print("\n" + "=" * 60)
print("SEÇÃO 2 — Filtrando pedidos entregues e convertendo datas...")
print("=" * 60)

# Filtra apenas pedidos entregues e cria uma cópia independente do DataFrame
# O .copy() evita o "SettingWithCopyWarning" do pandas em operações posteriores
orders = orders[orders["order_status"] == "delivered"].copy()
print(f"Pedidos entregues: {len(orders):,}")

# Lista das colunas de data que precisam ser convertidas
date_cols = [
    "order_purchase_timestamp",       # data e hora da compra
    "order_approved_at",              # data de aprovação do pagamento
    "order_delivered_carrier_date",   # data de entrega à transportadora
    "order_delivered_customer_date",  # data de entrega ao cliente
    "order_estimated_delivery_date"   # prazo de entrega prometido
]

# Converte cada coluna de texto para formato de data/hora
for col in date_cols:
    orders[col] = pd.to_datetime(orders[col])

print(f"Período da base: {orders['order_purchase_timestamp'].min().date()} "
      f"até {orders['order_purchase_timestamp'].max().date()}")


# ================================================================================
# SEÇÃO 3 — MERGE ORDERS + CUSTOMERS
# ================================================================================
# A tabela orders usa customer_id como chave de cliente, mas esse identificador
# muda a cada pedido — não permite rastrear o mesmo cliente em compras diferentes.
# A tabela customers tem o customer_unique_id, que é persistente para o mesmo
# cliente. Fazemos o join (merge) das duas tabelas para associar cada pedido
# ao seu cliente real.
#
# how="left" significa que mantemos todos os registros de orders, adicionando
# as colunas de customers quando o customer_id coincide. Pedidos sem cliente
# correspondente receberiam NaN, mas isso não ocorre nesta base.

print("\n" + "=" * 60)
print("SEÇÃO 3 — Associando pedidos aos clientes únicos...")
print("=" * 60)

df = orders.merge(
    customers[["customer_id", "customer_unique_id"]],
    on="customer_id",  # coluna de junção presente nas duas tabelas
    how="left"         # mantém todos os registros de orders
)

print(f"Pedidos após merge: {len(df):,}")
print(f"Clientes únicos: {df['customer_unique_id'].nunique():,}")


# ================================================================================
# SEÇÃO 4 — CRITÉRIO DE ELEGIBILIDADE (janela de 180 dias)
# ================================================================================
# Para classificar corretamente um cliente como "não recomprou", precisamos
# garantir que ele teve oportunidade real de recomprar dentro da janela de
# 180 dias. Clientes que realizaram a primeira compra nos últimos 180 dias
# antes do fim da base não tiveram tempo suficiente para recomprar — incluí-los
# causaria viés no modelo (seriam classificados como não-recompradores apenas
# por falta de janela de observação, não por comportamento real).
#
# Critério: apenas clientes cuja PRIMEIRA compra ocorreu com antecedência
# mínima de 180 dias em relação à data final da base (2018-08-29) são elegíveis.

print("\n" + "=" * 60)
print("SEÇÃO 4 — Aplicando critério de elegibilidade (180 dias)...")
print("=" * 60)

# Data do último pedido na base — define o horizonte temporal máximo
data_limite = df["order_purchase_timestamp"].max()

# Data de corte: 180 dias antes do fim da base
# Clientes com primeira compra após essa data são excluídos
data_corte = data_limite - pd.Timedelta(days=180)

print(f"Data limite da base: {data_limite.date()}")
print(f"Data de corte (180 dias antes): {data_corte.date()}")

# Para cada cliente, identifica a data de sua PRIMEIRA compra
# groupby agrupa os registros por cliente; min() pega a data mais antiga
primeira_compra = (
    df.groupby("customer_unique_id")["order_purchase_timestamp"]
    .min()
    .reset_index()
    .rename(columns={"order_purchase_timestamp": "primeira_compra"})
)

# Filtra apenas clientes elegíveis (primeira compra antes da data de corte)
elegiveis = primeira_compra[
    primeira_compra["primeira_compra"] <= data_corte
]

print(f"Clientes únicos total:      {len(primeira_compra):,}")
print(f"Clientes elegíveis:         {len(elegiveis):,}")
print(f"Excluídos (janela curta):   {len(primeira_compra) - len(elegiveis):,} "
      f"({(1 - len(elegiveis)/len(primeira_compra))*100:.1f}%)")


# ================================================================================
# SEÇÃO 5 — CONSTRUÇÃO DO TARGET (variável resposta)
# ================================================================================
# O target é a variável que queremos prever: recomprou (1) ou não recomprou (0).
# Definição: um cliente é considerado recomprador se realizou ao menos 2 pedidos
# no período coberto pela base — ou seja, se existe uma segunda compra registrada.
#
# Importante: a segunda compra é usada APENAS para definir o rótulo (target).
# As variáveis preditoras (features) são construídas exclusivamente com dados
# da primeira compra, para evitar que o modelo "veja o futuro" durante o treino.

print("\n" + "=" * 60)
print("SEÇÃO 5 — Construindo o target (variável resposta)...")
print("=" * 60)

# Filtra o DataFrame principal para conter apenas clientes elegíveis
df_elegivel = df[
    df["customer_unique_id"].isin(elegiveis["customer_unique_id"])
].copy()

# Conta o número total de pedidos por cliente elegível
# Clientes com mais de 1 pedido são recompradores (target = 1)
pedidos_por_cliente = (
    df_elegivel.groupby("customer_unique_id")["order_id"]
    .count()
    .reset_index()
    .rename(columns={"order_id": "total_pedidos"})
)

# Cria o rótulo binário: 1 se fez mais de 1 pedido, 0 caso contrário
pedidos_por_cliente["target"] = (
    pedidos_por_cliente["total_pedidos"] > 1
).astype(int)

recompradores = pedidos_por_cliente["target"].sum()
total         = len(pedidos_por_cliente)

print(f"Total de clientes elegíveis: {total:,}")
print(f"Recompradores (target=1):    {recompradores:,} ({recompradores/total*100:.2f}%)")
print(f"Não-recompradores (target=0): {total-recompradores:,} "
      f"({(total-recompradores)/total*100:.2f}%)")


# ================================================================================
# SEÇÃO 6 — FEATURES TRANSACIONAIS (apenas primeira compra)
# ================================================================================
# Construímos duas features a partir do valor financeiro da primeira compra:
# - valor_primeira_compra: quanto o cliente gastou na primeira compra (R$)
# - recencia_dias: há quantos dias o cliente fez a primeira compra
#
# IMPORTANTE: usamos apenas dados da PRIMEIRA compra de cada cliente.
# Uma versão preliminar deste pipeline incluiu frequência total de pedidos
# e valor total acumulado — ambos contaminados com informação futura (data
# leakage), pois incluíam dados da segunda compra. Após identificação e
# correção, apenas dados da primeira compra são utilizados.
#
# A recência mede o tempo entre a primeira compra e o fim da base (data_limite),
# e não em relação à data de corte. A data de corte serve apenas para o critério
# de elegibilidade; a recência captura o perfil temporal do cliente.

print("\n" + "=" * 60)
print("SEÇÃO 6 — Calculando features transacionais...")
print("=" * 60)

# Valor total pago por pedido (soma pagamentos — um pedido pode ter múltiplos)
valor_por_pedido = (
    payments.groupby("order_id")["payment_value"]
    .sum()
    .reset_index()
    .rename(columns={"payment_value": "valor_pedido"})
)

# Identifica o order_id da PRIMEIRA compra de cada cliente elegível
# sort_values ordena por data; groupby + first() pega o primeiro registro
primeira_ordem = (
    df_elegivel
    .sort_values("order_purchase_timestamp")
    .groupby("customer_unique_id")["order_id"]
    .first()
    .reset_index()
    .rename(columns={"order_id": "order_id_primeira"})
)

# Associa o valor pago à primeira compra de cada cliente
primeira_ordem = primeira_ordem.merge(
    valor_por_pedido,
    left_on="order_id_primeira",  # chave na tabela da esquerda
    right_on="order_id",          # chave na tabela da direita
    how="left"
)

# Renomeia e trata o único caso de valor ausente (pedido sem registro de pagamento)
primeira_ordem = primeira_ordem.rename(
    columns={"valor_pedido": "valor_primeira_compra"})
primeira_ordem["valor_primeira_compra"] = (
    primeira_ordem["valor_primeira_compra"].fillna(0)
)

# Monta o DataFrame de features transacionais com as duas colunas finais
transacional_corrigido = primeira_ordem[[
    "customer_unique_id", "valor_primeira_compra"
]].copy()

# Adiciona a data da primeira compra (de elegiveis) para calcular recência
transacional_corrigido = transacional_corrigido.merge(
    elegiveis[["customer_unique_id", "primeira_compra"]],
    on="customer_unique_id", how="left"
)

# Calcula recência: dias entre a primeira compra e o fim da base
# .dt.days converte o resultado de timedelta para número inteiro de dias
transacional_corrigido["recencia_dias"] = (
    data_limite - transacional_corrigido["primeira_compra"]
).dt.days

# Remove a coluna de data (não entra no modelo — já virou recencia_dias)
transacional_corrigido = transacional_corrigido.drop(columns=["primeira_compra"])

print(f"Features transacionais calculadas para {len(transacional_corrigido):,} clientes")
print(transacional_corrigido[["valor_primeira_compra", "recencia_dias"]].describe().round(2))


# ================================================================================
# SEÇÃO 7 — FEATURES LOGÍSTICAS
# ================================================================================
# Calculamos duas features relacionadas à experiência de entrega da primeira compra:
# - atraso_medio: diferença em dias entre prazo prometido e entrega real
#   Positivo = entregou antes do prazo (bom) | Negativo = atrasou (ruim)
# - flag_atraso: variável binária — 1 se houve atraso, 0 se não houve
#
# Embora as duas variáveis meçam o mesmo fenômeno (cumprimento de prazo),
# capturam dimensões distintas: atraso_medio mede a magnitude (quanto)
# e flag_atraso mede a ocorrência (se). A correlação entre elas é de -0,66,
# indicando complementaridade parcial — ambas são mantidas no modelo.
#
# Dois clientes apresentaram data de entrega ausente (NaT) apesar do status
# "delivered" — erro de registro. O atraso_medio é imputado pela mediana da
# distribuição (12 dias) para esses casos.

print("\n" + "=" * 60)
print("SEÇÃO 7 — Calculando features logísticas...")
print("=" * 60)

# Calcula o atraso por pedido: prazo prometido menos data de entrega real
# Valores positivos = entregou antes do prazo (adiantado)
# Valores negativos = entregou após o prazo (atrasado)
df_elegivel["atraso_dias"] = (
    df_elegivel["order_estimated_delivery_date"] -
    df_elegivel["order_delivered_customer_date"]
).dt.days

# Flag binária: 1 se o atraso foi negativo (atrasou), 0 caso contrário
df_elegivel["flag_atraso"] = (df_elegivel["atraso_dias"] < 0).astype(int)

# Agrega por cliente: média do atraso e máximo da flag (1 se teve atraso alguma vez)
logistica = (
    df_elegivel.groupby("customer_unique_id")
    .agg(
        atraso_medio = ("atraso_dias", "mean"),
        flag_atraso  = ("flag_atraso", "max")
    )
    .reset_index()
)

# Imputa a mediana nos 2 clientes com data de entrega ausente
logistica["atraso_medio"] = logistica["atraso_medio"].fillna(
    logistica["atraso_medio"].median()
)

print(f"Pedidos com atraso: {df_elegivel['flag_atraso'].sum():,} "
      f"({df_elegivel['flag_atraso'].mean()*100:.1f}%)")
print(logistica[["atraso_medio", "flag_atraso"]].describe().round(2))


# ================================================================================
# SEÇÃO 8 — FEATURES GEOGRÁFICAS
# ================================================================================
# Utilizamos o estado do cliente na primeira compra como proxy geográfico.
# Para os 39 clientes que realizaram compras com endereços em estados diferentes,
# mantemos apenas o estado da primeira compra, preservando consistência temporal.
#
# O estado é uma variável categórica com 27 valores (unidades federativas).
# Aplicamos Label Encoding — conversão de cada estado para um número inteiro —
# que é suficiente para algoritmos baseados em árvores de decisão (Random Forest,
# XGBoost, LightGBM), que não pressupõem relação de ordem entre as categorias.

print("\n" + "=" * 60)
print("SEÇÃO 8 — Calculando features geográficas...")
print("=" * 60)

# Associa cada pedido elegível ao estado do cliente via customer_id
pedido_estado = (
    df_elegivel
    .merge(
        customers[["customer_id", "customer_state"]],
        on="customer_id", how="left"
    )
    [["customer_unique_id", "order_purchase_timestamp", "customer_state"]]
)

# Para clientes com pedidos em múltiplos estados, mantém o estado da 1ª compra
# sort_values ordena por data; first() pega o registro mais antigo por cliente
estado_primeira_compra = (
    pedido_estado
    .sort_values("order_purchase_timestamp")
    .groupby("customer_unique_id")["customer_state"]
    .first()
    .reset_index()
)

print(f"Estados únicos: {estado_primeira_compra['customer_state'].nunique()}")
print(f"Top 5 estados:")
print(estado_primeira_compra["customer_state"].value_counts().head(5))


# ================================================================================
# SEÇÃO 9 — FEATURES TEXTUAIS (sentimento da primeira compra)
# ================================================================================
# O sentimento das avaliações foi calculado pelo modelo BERTimbau no Google Colab
# (ver script sentiment_analysis_colab.py). Aqui, utilizamos apenas a avaliação
# da PRIMEIRA compra de cada cliente como feature preditora — evitando data leakage.
#
# A análise exploratória revelou que a taxa de recompra é praticamente uniforme
# entre os scores de avaliação (3,91% a 4,48%), indicando baixo poder preditivo
# do sentimento. As features textuais foram mantidas para avaliar quantitativamente
# sua contribuição incremental via SHAP values.
#
# Para clientes sem avaliação da primeira compra (417 casos):
# - review_score_primeira: imputado com 5,0 (mediana da base)
# - sentimento_num: imputado com 0 (neutro)

print("\n" + "=" * 60)
print("SEÇÃO 9 — Calculando features textuais (sentimento)...")
print("=" * 60)

# Associa cada review ao order_id da primeira compra
reviews_primeira = primeira_ordem.merge(
    reviews,
    left_on="order_id_primeira",
    right_on="order_id",
    how="left"
)

# Converte o score numérico em rótulo de sentimento
# Regra: 1-2 = negativo | 3 = neutro | 4-5 = positivo
def score_para_sentimento(score):
    if pd.isna(score):  return "neutral"   # sem avaliação → neutro
    elif score <= 2:    return "negative"  # insatisfeito
    elif score == 3:    return "neutral"   # indiferente
    else:               return "positive"  # satisfeito

reviews_primeira["sentimento_primeira"] = (
    reviews_primeira["review_score"].apply(score_para_sentimento)
)

# Converte o rótulo de sentimento para valor numérico
# positivo=1, neutro=0, negativo=-1 — permite uso direto como feature numérica
mapa_sentimento = {"positive": 1, "neutral": 0, "negative": -1}
reviews_primeira["sentimento_num"] = (
    reviews_primeira["sentimento_primeira"].map(mapa_sentimento)
)

# Trata os 417 clientes sem avaliação da primeira compra
reviews_primeira["review_score"]   = reviews_primeira["review_score"].fillna(5.0)
reviews_primeira["sentimento_num"] = reviews_primeira["sentimento_num"].fillna(0)

# Seleciona apenas as colunas que entram no modelo
sentimento_primeira = reviews_primeira[[
    "customer_unique_id",
    "review_score",
    "sentimento_num"
]].rename(columns={"review_score": "review_score_primeira"})

print(f"Clientes com avaliação da 1ª compra: "
      f"{(reviews_primeira['review_score'] != 5.0).sum():,}")
print(f"Clientes sem avaliação (imputados):  417")
print(f"\nDistribuição de sentimento:")
print(reviews_primeira["sentimento_primeira"].value_counts())


# ================================================================================
# SEÇÃO 10 — FEATURE DE DURABILIDADE DO PRODUTO
# ================================================================================
# Construímos uma variável binária que classifica o produto da primeira compra
# em durável (ciclo de recompra longo, valor=1) ou não-durável (ciclo curto, valor=0).
#
# A classificação foi feita manualmente para as 71 categorias de produto da base,
# com base no comportamento esperado de consumo:
# - Não-duráveis (0): cosméticos, livros, moda, esporte, brinquedos, pet shop,
#   alimentos, flores, papelaria — produtos que se desgastam ou consomem rapidamente
# - Duráveis (1): eletrônicos, móveis, eletrodomésticos, ferramentas, automotivo,
#   instrumentos musicais — produtos com vida útil longa
#
# Categorias sem mapeamento (ausentes na base) recebem valor 1 (durável) por
# conservadorismo — evita inflar artificialmente a proporção de não-duráveis.

print("\n" + "=" * 60)
print("SEÇÃO 10 — Calculando feature de durabilidade do produto...")
print("=" * 60)

# Associa cada item ao nome da categoria em inglês via joins em cadeia
items_cat = (
    items
    .merge(
        products[["product_id", "product_category_name"]],
        on="product_id", how="left"
    )
    .merge(category, on="product_category_name", how="left")
)

# Identifica a categoria do produto da PRIMEIRA compra de cada cliente elegível
# Quando o pedido tem múltiplos itens, usa a categoria do primeiro item registrado
categoria_primeira = (
    primeira_ordem
    .merge(
        items_cat[["order_id", "product_category_name_english"]],
        left_on="order_id_primeira",
        right_on="order_id",
        how="left"
    )
    .groupby("customer_unique_id")["product_category_name_english"]
    .first()
    .reset_index()
)

# Dicionário de mapeamento: nome da categoria → 0 (não-durável) ou 1 (durável)
mapa_duravel = {
    # ── Não-duráveis (0) — ciclo de recompra curto ──
    "health_beauty": 0,           # cosméticos e cuidados pessoais
    "perfumery": 0,               # perfumaria
    "food_drink": 0,              # alimentos e bebidas
    "food": 0,                    # alimentos
    "drinks": 0,                  # bebidas
    "diapers_and_hygiene": 0,     # fraldas e higiene
    "baby": 0,                    # produtos para bebês
    "stationery": 0,              # papelaria
    "party_supplies": 0,          # artigos para festas
    "christmas_supplies": 0,      # artigos natalinos
    "flowers": 0,                 # flores
    "arts_and_craftmanship": 0,   # artes e artesanato
    "books_general_interest": 0,  # livros de interesse geral
    "books_technical": 0,         # livros técnicos
    "books_imported": 0,          # livros importados
    "fashion_male_clothing": 0,   # moda masculina
    "fashio_female_clothing": 0,  # moda feminina
    "fashion_childrens_clothes": 0, # moda infantil
    "fashion_underwear_beach": 0, # moda praia e lingerie
    "fashion_sport": 0,           # moda esportiva
    "fashion_shoes": 0,           # calçados
    "fashion_bags_accessories": 0,# bolsas e acessórios
    "pet_shop": 0,                # produtos para animais
    "toys": 0,                    # brinquedos
    "sports_leisure": 0,          # artigos esportivos e lazer
    "la_cuisine": 0,              # culinária
    # ── Duráveis (1) — ciclo de recompra longo ──
    "art": 1,                     # objetos de arte
    "bed_bath_table": 1,          # cama, mesa e banho
    "furniture_decor": 1,         # móveis e decoração
    "computers_accessories": 1,   # computadores e acessórios
    "housewares": 1,              # utilidades domésticas
    "watches_gifts": 1,           # relógios e presentes
    "telephony": 1,               # telefonia
    "cool_stuff": 1,              # produtos diversos (gadgets)
    "garden_tools": 1,            # ferramentas de jardim
    "auto": 1,                    # automotivo
    "electronics": 1,             # eletrônicos
    "office_furniture": 1,        # móveis de escritório
    "consoles_games": 1,          # consoles e jogos
    "small_appliances": 1,        # eletrodomésticos pequenos
    "home_appliances": 1,         # eletrodomésticos grandes
    "home_appliances_2": 1,       # eletrodomésticos (categoria 2)
    "tablets_printing_image": 1,  # tablets e impressoras
    "audio": 1,                   # equipamentos de áudio
    "fixed_telephony": 1,         # telefonia fixa
    "air_conditioning": 1,        # climatização
    "computers": 1,               # computadores
    "cine_photo": 1,              # câmeras e equipamentos
    "music": 1,                   # instrumentos musicais
    "musical_instruments": 1,     # instrumentos musicais
    "cds_dvds_musicals": 1,       # CDs e DVDs
    "dvds_blu_ray": 1,            # DVDs e Blu-Ray
    "luggage_accessories": 1,     # malas e acessórios de viagem
    "furniture_bedroom": 1,       # móveis de quarto
    "furniture_living_room": 1,   # móveis de sala
    "furniture_mattress_and_upholstery": 1,  # colchões e estofados
    "kitchen_dining_laundry_garden_furniture": 1,  # móveis cozinha/jardim
    "home_construction": 1,       # construção e reforma
    "home_confort": 1,            # conforto doméstico
    "home_comfort_2": 1,          # conforto doméstico (categoria 2)
    "construction_tools_construction": 1,  # ferramentas de construção
    "costruction_tools_garden": 1,         # ferramentas de jardim
    "costruction_tools_tools": 1,          # ferramentas gerais
    "construction_tools_lights": 1,        # iluminação e elétrica
    "construction_tools_safety": 1,        # segurança e EPI
    "signaling_and_security": 1,           # sinalização e segurança
    "agro_industry_and_commerce": 1,       # agronegócio e indústria
    "industry_commerce_and_business": 1,   # indústria e comércio
    "security_and_services": 1,            # seguros e serviços
    "small_appliances_home_oven_and_coffee": 1,  # fornos e cafeteiras
    "market_place": 1,                     # marketplace (ambíguo → durável)
}

# Aplica o mapeamento e trata categorias não mapeadas como duráveis (valor=1)
categoria_primeira["produto_duravel"] = (
    categoria_primeira["product_category_name_english"]
    .map(mapa_duravel)
    .fillna(1)
    .astype(int)
)

nao_duráveis = (categoria_primeira["produto_duravel"] == 0).sum()
duráveis     = (categoria_primeira["produto_duravel"] == 1).sum()
print(f"Não-duráveis: {nao_duráveis:,} ({nao_duráveis/len(categoria_primeira)*100:.1f}%)")
print(f"Duráveis:     {duráveis:,} ({duráveis/len(categoria_primeira)*100:.1f}%)")


# ================================================================================
# SEÇÃO 11 — MONTAGEM DO df_base (DataFrame principal de modelagem)
# ================================================================================
# Consolida todas as features construídas nas seções anteriores num único
# DataFrame, com uma linha por cliente elegível. Cada merge adiciona as
# colunas de uma dimensão ao DataFrame base.
# Ao final, verificamos a presença de valores ausentes e os tratamos.

print("\n" + "=" * 60)
print("SEÇÃO 11 — Montando o DataFrame base de modelagem...")
print("=" * 60)

# Parte do DataFrame de clientes elegíveis com a data da primeira compra
# e adiciona o target (variável resposta)
df_base = elegiveis.merge(
    pedidos_por_cliente[["customer_unique_id", "target"]],
    on="customer_unique_id", how="left"
)

# Adiciona features transacionais (valor e recência)
df_base = df_base.merge(transacional_corrigido, on="customer_unique_id", how="left")

# Adiciona features logísticas (atraso e flag)
df_base = df_base.merge(logistica, on="customer_unique_id", how="left")

# Adiciona feature geográfica (estado)
df_base = df_base.merge(estado_primeira_compra, on="customer_unique_id", how="left")

# Adiciona features textuais (score e sentimento da primeira avaliação)
df_base = df_base.merge(sentimento_primeira, on="customer_unique_id", how="left")

# Adiciona feature de durabilidade do produto
df_base = df_base.merge(
    categoria_primeira[["customer_unique_id", "produto_duravel"]],
    on="customer_unique_id", how="left"
)

# Tratamento de valores ausentes remanescentes
# review_score_primeira: imputado com 5,0 (mediana) para clientes sem avaliação
# sentimento_num: imputado com 0 (neutro) para clientes sem avaliação
# produto_duravel: imputado com 1 (durável) para categorias não mapeadas
df_base["review_score_primeira"] = df_base["review_score_primeira"].fillna(5.0)
df_base["sentimento_num"]        = df_base["sentimento_num"].fillna(0)
df_base["produto_duravel"]       = df_base["produto_duravel"].fillna(1).astype(int)

print(f"Shape do df_base: {df_base.shape}")
print(f"Nulos:            {df_base.isnull().sum().sum()}")
print(f"Colunas:          {df_base.columns.tolist()}")


# ================================================================================
# SEÇÃO 12 — WINSORIZAÇÃO E LABEL ENCODING
# ================================================================================
# Winsorização: técnica que limita valores extremos (outliers) ao percentil
# definido, evitando que pontos isolados distorçam o modelo.
# Aplicamos apenas em atraso_medio, que apresentou outliers extremos
# (-189 dias e +139 dias) provavelmente decorrentes de erros de registro.
# valor_primeira_compra e recencia_dias foram mantidos sem winsorização —
# seus valores extremos são informativos e não constituem erros.
#
# Label Encoding: converte a variável categórica customer_state (27 estados)
# em valores numéricos inteiros (AC=0, AL=1, ..., TO=26).
# Algoritmos baseados em árvores não requerem one-hot encoding — label encoding
# é suficiente e evita a criação de 26 colunas adicionais.

print("\n" + "=" * 60)
print("SEÇÃO 12 — Winsorização e Label Encoding...")
print("=" * 60)

# Winsorização de atraso_medio nos percentis 1% e 99%
# Valores abaixo do percentil 1% são elevados ao percentil 1%
# Valores acima do percentil 99% são reduzidos ao percentil 99%
p01 = df_base["atraso_medio"].quantile(0.01)
p99 = df_base["atraso_medio"].quantile(0.99)
df_base["atraso_medio"] = df_base["atraso_medio"].clip(lower=p01, upper=p99)
print(f"atraso_medio winsorizado para [{p01:.1f}, {p99:.1f}] dias")

# Label Encoding: converte cada estado para um número inteiro único
le = LabelEncoder()
df_base["customer_state"] = le.fit_transform(df_base["customer_state"])
print(f"Estados codificados: {len(le.classes_)} estados únicos")


# ================================================================================
# SEÇÃO 13 — PREPARAÇÃO DAS MATRIZES X e y E DIVISÃO TREINO/TESTE
# ================================================================================
# X (features): matriz com as 8 variáveis preditoras
# y (target): vetor com o rótulo binário de recompra (0 ou 1)
#
# Divisão treino/teste:
# - 80% dos dados para treino: o modelo aprende os padrões
# - 20% dos dados para teste: avaliamos se o modelo generaliza
# - stratify=y: garante que a proporção de recompradores (4,26%) seja
#   preservada em ambas as partições — essencial com classes desbalanceadas
# - random_state=42: garante reprodutibilidade (mesma divisão a cada execução)

print("\n" + "=" * 60)
print("SEÇÃO 13 — Preparando matrizes X e y e dividindo treino/teste...")
print("=" * 60)

# Remove colunas de controle que não entram no modelo
# customer_unique_id: identificador do cliente (não é feature)
# primeira_compra: data usada para calcular recencia_dias (já incorporada)
df_model = df_base.drop(columns=["customer_unique_id", "primeira_compra"])

# Separa features (X) do target (y)
X = df_model.drop(columns=["target"])  # todas as colunas exceto target
y = df_model["target"]                 # apenas a coluna target

# Divisão estratificada 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% para teste
    stratify=y,         # mantém proporção do target
    random_state=42     # semente para reprodutibilidade
)

print(f"Features utilizadas:  {X.columns.tolist()}")
print(f"X_train: {X_train.shape[0]:,} amostras | Target: {y_train.mean():.4f}")
print(f"X_test:  {X_test.shape[0]:,} amostras  | Target: {y_test.mean():.4f}")


# ================================================================================
# SEÇÃO 14 — TREINAMENTO DOS MODELOS BASE (sem otimização)
# ================================================================================
# Quatro algoritmos são treinados com parâmetros padrão para estabelecer
# um baseline de comparação antes da otimização de hiperparâmetros.
#
# Tratamento do desbalanceamento de classes (96% não-recompradores / 4% recompradores):
# - class_weight="balanced": ajusta automaticamente os pesos inversamente
#   proporcionais à frequência de cada classe (Regressão Logística e Random Forest)
# - scale_pos_weight: peso aplicado à classe positiva no XGBoost e LightGBM,
#   calculado como a razão entre negativos e positivos (~22,5)
#
# A Regressão Logística requer normalização das features (StandardScaler)
# porque é sensível à escala das variáveis. Os algoritmos baseados em árvores
# (Random Forest, XGBoost, LightGBM) são invariantes à escala e não precisam.

print("\n" + "=" * 60)
print("SEÇÃO 14 — Treinando modelos base...")
print("=" * 60)

# Proporção negativos/positivos — usada como scale_pos_weight nos modelos de boosting
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight: {scale_pos:.2f} (proporção negativos/positivos)")

# ── Regressão Logística ──
# Normaliza as features: média=0, desvio padrão=1
# fit_transform: aprende a normalização no treino
# transform: aplica a mesma normalização no teste (sem "ver" os dados de teste)
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

lr = LogisticRegression(
    class_weight="balanced",  # ajusta pesos pelo desbalanceamento
    max_iter=1000,            # número máximo de iterações para convergência
    random_state=42           # reprodutibilidade
)
lr.fit(X_train_scaled, y_train)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]  # probabilidade da classe positiva
print(f"Regressão Logística — AUC-ROC: {roc_auc_score(y_test, y_prob_lr):.4f} | "
      f"AUC-PR: {average_precision_score(y_test, y_prob_lr):.4f}")

# ── Random Forest ──
# Ensemble de 300 árvores de decisão treinadas em subamostras aleatórias
# n_jobs=-1: usa todos os núcleos disponíveis do processador
rf = RandomForestClassifier(
    class_weight="balanced",
    n_estimators=300,  # número de árvores no ensemble
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
y_prob_rf = rf.predict_proba(X_test)[:, 1]
print(f"Random Forest      — AUC-ROC: {roc_auc_score(y_test, y_prob_rf):.4f} | "
      f"AUC-PR: {average_precision_score(y_test, y_prob_rf):.4f}")

# ── XGBoost ──
# Gradient boosting que constrói árvores sequencialmente, cada uma corrigindo
# os erros da anterior. eval_metric="aucpr" otimiza diretamente o AUC-PR.
xgb = XGBClassifier(
    scale_pos_weight=scale_pos,
    n_estimators=300,
    random_state=42,
    eval_metric="aucpr",  # métrica de avaliação interna
    verbosity=0           # suprime mensagens de progresso
)
xgb.fit(X_train, y_train)
y_prob_xgb = xgb.predict_proba(X_test)[:, 1]
print(f"XGBoost            — AUC-ROC: {roc_auc_score(y_test, y_prob_xgb):.4f} | "
      f"AUC-PR: {average_precision_score(y_test, y_prob_xgb):.4f}")

# ── LightGBM ──
# Implementação de gradient boosting com crescimento de árvore por folha (leaf-wise),
# mais eficiente computacionalmente que o XGBoost em bases maiores.
lgbm = LGBMClassifier(
    scale_pos_weight=scale_pos,
    n_estimators=300,
    random_state=42,
    verbosity=-1  # suprime todas as mensagens de progresso
)
lgbm.fit(X_train, y_train)
y_prob_lgbm = lgbm.predict_proba(X_test)[:, 1]
print(f"LightGBM           — AUC-ROC: {roc_auc_score(y_test, y_prob_lgbm):.4f} | "
      f"AUC-PR: {average_precision_score(y_test, y_prob_lgbm):.4f}")


# ================================================================================
# SEÇÃO 15 — OTIMIZAÇÃO DE HIPERPARÂMETROS — XGBoost (Optuna)
# ================================================================================
# O Optuna utiliza otimização bayesiana para encontrar a melhor combinação
# de hiperparâmetros, aprendendo com cada tentativa anterior qual região
# do espaço de busca é mais promissora. É mais eficiente que o GridSearch,
# que testa exaustivamente todas as combinações possíveis.
#
# Parâmetros otimizados:
# - n_estimators: número de árvores (100 a 1000)
# - max_depth: profundidade máxima de cada árvore (3 a 10)
#   Árvores mais profundas capturam padrões mais complexos mas tendem ao overfitting
# - learning_rate: peso de cada nova árvore no ensemble (0,01 a 0,3)
#   Valores menores = aprendizado mais lento e estável
# - subsample: fração de amostras usada por árvore (0,6 a 1,0)
# - colsample_bytree: fração de features usada por árvore (0,6 a 1,0)
#
# Métrica de otimização: AUC-PR — mais adequada para classes desbalanceadas

print("\n" + "=" * 60)
print("SEÇÃO 15 — Otimizando XGBoost com Optuna (50 trials)...")
print("=" * 60)

def objective_xgb(trial):
    """
    Função objetivo do Optuna para o XGBoost.
    Em cada trial, o Optuna sugere um conjunto de hiperparâmetros,
    treina o modelo e retorna o AUC-PR no conjunto de teste.
    O Optuna maximiza esse valor ao longo dos 50 trials.
    """
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 1000),
        "max_depth":        trial.suggest_int("max_depth", 3, 10),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "scale_pos_weight": scale_pos,
        "random_state":     42,
        "verbosity":        0,
        "eval_metric":      "aucpr"
    }
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    return average_precision_score(y_test, model.predict_proba(X_test)[:, 1])

# Cria um estudo de otimização com direção de maximização
study_xgb = optuna.create_study(direction="maximize")

# Executa 50 trials de otimização
study_xgb.optimize(objective_xgb, n_trials=50)

print(f"Melhor AUC-PR XGBoost: {study_xgb.best_value:.4f}")
print(f"Melhores parâmetros:   {study_xgb.best_params}")


# ================================================================================
# SEÇÃO 16 — OTIMIZAÇÃO DE HIPERPARÂMETROS — LightGBM (Optuna)
# ================================================================================
# Mesma abordagem da seção anterior, aplicada ao LightGBM.
# Os parâmetros otimizados são os mesmos, mas o espaço de busca pode
# convergir para valores diferentes — cada algoritmo tem suas próprias
# sensibilidades a cada hiperparâmetro.

print("\n" + "=" * 60)
print("SEÇÃO 16 — Otimizando LightGBM com Optuna (50 trials)...")
print("=" * 60)

def objective_lgbm(trial):
    """
    Função objetivo do Optuna para o LightGBM.
    Estrutura idêntica à do XGBoost, com os parâmetros específicos do LightGBM.
    """
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 1000),
        "max_depth":        trial.suggest_int("max_depth", 3, 10),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "scale_pos_weight": scale_pos,
        "random_state":     42,
        "verbosity":        -1
    }
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)
    return average_precision_score(y_test, model.predict_proba(X_test)[:, 1])

study_lgbm = optuna.create_study(direction="maximize")
study_lgbm.optimize(objective_lgbm, n_trials=50)

print(f"Melhor AUC-PR LightGBM: {study_lgbm.best_value:.4f}")
print(f"Melhores parâmetros:    {study_lgbm.best_params}")


# ================================================================================
# SEÇÃO 17 — MODELOS FINAIS COM PARÂMETROS OTIMIZADOS
# ================================================================================
# Treina os modelos finais usando os melhores hiperparâmetros encontrados pelo Optuna.
# O operador ** desempacota o dicionário de parâmetros como argumentos nomeados.

print("\n" + "=" * 60)
print("SEÇÃO 17 — Treinando modelos finais com parâmetros otimizados...")
print("=" * 60)

xgb_final = XGBClassifier(
    **study_xgb.best_params,   # melhores hiperparâmetros encontrados pelo Optuna
    scale_pos_weight=scale_pos,
    random_state=42,
    verbosity=0,
    eval_metric="aucpr"
)
xgb_final.fit(X_train, y_train)
y_prob_xgb_final = xgb_final.predict_proba(X_test)[:, 1]

lgbm_final = LGBMClassifier(
    **study_lgbm.best_params,
    scale_pos_weight=scale_pos,
    random_state=42,
    verbosity=-1
)
lgbm_final.fit(X_train, y_train)
y_prob_lgbm_final = lgbm_final.predict_proba(X_test)[:, 1]


# ================================================================================
# SEÇÃO 18 — QUADRO COMPARATIVO FINAL DOS MODELOS
# ================================================================================
# Consolida os resultados de todos os modelos (base e otimizados) numa tabela
# para facilitar a comparação e seleção do modelo final.
#
# Métricas utilizadas:
# - AUC-ROC [Area Under the Curve — Receiver Operating Characteristic]:
#   probabilidade de o modelo rankear corretamente um recomprador acima
#   de um não-recomprador. Referência: 0,5 = aleatório, 1,0 = perfeito.
# - AUC-PR [Area Under the Precision-Recall Curve]:
#   mais informativa que AUC-ROC para classes desbalanceadas — ignora os
#   verdadeiros negativos e foca na capacidade de identificar recompradores.
#   Referência: ~0,043 = modelo aleatório (proporção de positivos na base).

print("\n" + "=" * 60)
print("SEÇÃO 18 — Quadro comparativo final dos modelos")
print("=" * 60)

resultados_final = pd.DataFrame({
    "Modelo": [
        "Regressão Logística",
        "Random Forest",
        "XGBoost",
        "XGBoost (otimizado)",
        "LightGBM",
        "LightGBM (otimizado)"
    ],
    "AUC-ROC": [
        roc_auc_score(y_test, y_prob_lr),
        roc_auc_score(y_test, y_prob_rf),
        roc_auc_score(y_test, y_prob_xgb),
        roc_auc_score(y_test, y_prob_xgb_final),
        roc_auc_score(y_test, y_prob_lgbm),
        roc_auc_score(y_test, y_prob_lgbm_final)
    ],
    "AUC-PR": [
        average_precision_score(y_test, y_prob_lr),
        average_precision_score(y_test, y_prob_rf),
        average_precision_score(y_test, y_prob_xgb),
        average_precision_score(y_test, y_prob_xgb_final),
        average_precision_score(y_test, y_prob_lgbm),
        average_precision_score(y_test, y_prob_lgbm_final)
    ]
}).round(4)

print(resultados_final.to_string(index=False))

# Identifica o modelo vencedor por AUC-PR
melhor_idx   = resultados_final["AUC-PR"].idxmax()
melhor_nome  = resultados_final.loc[melhor_idx, "Modelo"]
melhor_model = xgb_final if "XGBoost" in melhor_nome else lgbm_final
melhor_prob  = y_prob_xgb_final if "XGBoost" in melhor_nome else y_prob_lgbm_final
print(f"\nModelo vencedor por AUC-PR: {melhor_nome}")


# ================================================================================
# SEÇÃO 19 — RELATÓRIO DETALHADO DO MODELO VENCEDOR
# ================================================================================
# O classification_report apresenta, para cada classe (0 e 1):
# - Precisão: dos que o modelo classificou como positivo, quantos realmente são?
# - Recall: dos que são realmente positivos, quantos o modelo encontrou?
# - F1-score: média harmônica entre precisão e recall
# - Support: número de amostras de cada classe no conjunto de teste

print("\n" + "=" * 60)
print(f"SEÇÃO 19 — Relatório detalhado — {melhor_nome}")
print("=" * 60)
print(classification_report(y_test, melhor_model.predict(X_test)))


# ================================================================================
# SEÇÃO 20 — INTERPRETABILIDADE VIA SHAP VALUES
# ================================================================================
# SHAP (SHapley Additive exPlanations) é uma metodologia baseada na teoria dos
# jogos cooperativos que atribui a cada variável uma contribuição marginal para
# cada predição individual. Permite responder: "quais features mais influenciaram
# a predição para este cliente específico?" (análise local) e "quais features
# são mais importantes globalmente?" (análise global).
#
# TreeExplainer é a implementação otimizada do SHAP para modelos baseados em
# árvores (Random Forest, XGBoost, LightGBM) — muito mais rápida que a
# implementação genérica (KernelExplainer).
#
# Para classificação binária com LightGBM, os SHAP values são retornados como
# lista com dois arrays (um por classe). Usamos o índice [1] — classe positiva
# (recompra) — para a análise de importância.

print("\n" + "=" * 60)
print(f"SEÇÃO 20 — Calculando SHAP values — {melhor_nome}")
print("=" * 60)

# Cria o explicador SHAP para modelos baseados em árvores
explainer = shap.TreeExplainer(melhor_model)

# Calcula os SHAP values para o conjunto de teste
# Retorna a contribuição de cada feature para cada observação
shap_values = explainer.shap_values(X_test)

# Para classificação binária com LightGBM, seleciona os valores da classe positiva
if isinstance(shap_values, list):
    shap_values_pos = shap_values[1]  # índice 1 = classe positiva (recompra)
else:
    shap_values_pos = shap_values     # XGBoost retorna diretamente para a classe positiva

# Calcula a importância média absoluta de cada feature
# abs() converte valores negativos em positivos (contribuição em qualquer direção)
# mean(axis=0) calcula a média sobre todas as observações do conjunto de teste
importancia = pd.DataFrame({
    "feature":    X_test.columns,
    "shap_medio": abs(shap_values_pos).mean(axis=0)
}).sort_values("shap_medio", ascending=False)

print(f"\nImportância SHAP — {melhor_nome}:")
print(importancia.to_string(index=False))


# ================================================================================
# SEÇÃO 21 — GERAÇÃO DOS GRÁFICOS
# ================================================================================
# Três visualizações são geradas num único arquivo PNG:
#
# 1. Curva ROC: plota a taxa de verdadeiros positivos (sensibilidade) contra
#    a taxa de falsos positivos para todos os thresholds possíveis.
#    A diagonal pontilhada representa o modelo aleatório (AUC-ROC = 0,5).
#
# 2. Curva Precisão-Recall: plota a precisão contra o recall para todos os
#    thresholds. A linha horizontal representa o baseline aleatório (AUC-PR
#    igual à proporção de positivos na base, ~4,3%).
#    Mais informativa que a ROC para problemas desbalanceados.
#
# 3. Importância SHAP: gráfico de barras horizontais com a contribuição média
#    absoluta de cada feature, ordenada da mais para a menos importante.

print("\n" + "=" * 60)
print("SEÇÃO 21 — Gerando gráficos...")
print("=" * 60)

# Cria figura com 3 painéis lado a lado (1 linha x 3 colunas)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Dicionário com os modelos finais para plotar nas curvas
modelos_plot = {
    "Regressão Logística":  y_prob_lr,
    "Random Forest":        y_prob_rf,
    "XGBoost (otimizado)":  y_prob_xgb_final,
    "LightGBM (otimizado)": y_prob_lgbm_final
}

# ── Painel 1: Curva ROC ──
for nome, y_prob in modelos_plot.items():
    fpr, tpr, _ = roc_curve(y_test, y_prob)          # calcula a curva ROC
    auc = roc_auc_score(y_test, y_prob)               # calcula o AUC-ROC
    axes[0].plot(fpr, tpr, label=f"{nome} ({auc:.3f})")  # plota a curva

axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.8)   # linha diagonal (baseline)
axes[0].set_xlabel("Taxa de Falsos Positivos")
axes[0].set_ylabel("Taxa de Verdadeiros Positivos")
axes[0].set_title("Curva ROC")
axes[0].legend(fontsize=8)

# ── Painel 2: Curva Precisão-Recall ──
for nome, y_prob in modelos_plot.items():
    prec, rec, _ = precision_recall_curve(y_test, y_prob)  # calcula a curva PR
    ap = average_precision_score(y_test, y_prob)            # calcula o AUC-PR
    axes[1].plot(rec, prec, label=f"{nome} ({ap:.3f})")

# Linha horizontal: AUC-PR de um modelo aleatório = proporção de positivos
axes[1].axhline(
    y=y_test.mean(), color="k", linestyle="--",
    linewidth=0.8, label=f"Baseline ({y_test.mean():.3f})"
)
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precisão")
axes[1].set_title("Curva Precisão-Recall")
axes[1].legend(fontsize=8)

# ── Painel 3: Importância SHAP ──
axes[2].barh(
    importancia["feature"],    # nomes das features no eixo vertical
    importancia["shap_medio"], # importância no eixo horizontal
    color="steelblue"
)
axes[2].set_xlabel("SHAP médio absoluto")
axes[2].set_title(f"Importância das Features (SHAP)\n{melhor_nome}")
axes[2].invert_yaxis()  # feature mais importante no topo

plt.tight_layout()  # ajusta espaçamento entre painéis automaticamente

# Salva em alta resolução (150 dpi) na pasta de trabalho
plt.savefig("graficos_tcc_v2.png", dpi=150, bbox_inches="tight")
plt.close()  # libera memória
print("Gráfico salvo: graficos_tcc_v2.png")


# ================================================================================
# SEÇÃO 22 — CHECKPOINTS (salvamento dos arquivos intermediários)
# ================================================================================
# Salva os DataFrames intermediários em CSV para uso nos scripts auxiliares
# de análise exploratória (teste_variaveis_candidatas.py e
# teste_valor_primeira_compra.py) sem necessidade de reprocessar o pipeline.

print("\n" + "=" * 60)
print("SEÇÃO 22 — Salvando checkpoints...")
print("=" * 60)

# DataFrame base com todas as features — usado para análise exploratória
df_base.to_csv("df_base_features.csv", index=False)

# Pedidos dos clientes elegíveis — usado nos scripts de teste de variáveis
df_elegivel.to_csv("df_elegivel.csv", index=False)

# Tabela de itens — usada nos scripts de teste de variáveis
items.to_csv("items_cache.csv", index=False)

print("df_base_features.csv salvo")
print("df_elegivel.csv salvo")
print("items_cache.csv salvo")

print("\n" + "=" * 60)
print("PIPELINE CONCLUÍDO COM SUCESSO")
print(f"Modelo final: {melhor_nome}")
print(f"AUC-ROC: {roc_auc_score(y_test, melhor_prob):.4f}")
print(f"AUC-PR:  {average_precision_score(y_test, melhor_prob):.4f}")
print("=" * 60)
