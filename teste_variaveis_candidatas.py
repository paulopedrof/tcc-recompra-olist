"""
TESTE DE VARIÁVEIS CANDIDATAS — FRETE E TEMPO DE POSTAGEM
Objetivo: avaliar o poder preditivo do valor do frete e do tempo de postagem
do vendedor como features adicionais para o modelo de recompra.
Resultado esperado: inclusão ou descarte justificado no pipeline final.
"""

import pandas as pd

# ── Carrega dados necessários ──
# (rodar após tcc_completo.py ou carregar dos CSVs intermediários)
try:
    _ = df_elegivel
    _ = items
    print("Variáveis carregadas da sessão atual.")
except NameError:
    df_elegivel = pd.read_csv("df_elegivel.csv")
    df_elegivel["order_approved_at"] = pd.to_datetime(
        df_elegivel["order_approved_at"])
    df_elegivel["order_delivered_carrier_date"] = pd.to_datetime(
        df_elegivel["order_delivered_carrier_date"])
    items = pd.read_csv("items_cache.csv")
    print("Variáveis carregadas dos CSVs intermediários.")

# ── 1. Frete total por pedido ──
# Soma o frete de todos os itens do pedido (pode haver múltiplos vendedores)
items_frete = (items.groupby("order_id")["freight_value"]
               .sum().reset_index()
               .rename(columns={"freight_value": "frete_total"}))

print(f"\nFrete total por pedido:")
print(items_frete["frete_total"].describe().round(2))

# ── 2. Tempo de postagem por pedido ──
# Intervalo entre aprovação do pagamento e entrega à transportadora
# Mede a agilidade do vendedor em despachar o produto
df_postagem = df_elegivel[[
    "order_id", "customer_unique_id",
    "order_approved_at",
    "order_delivered_carrier_date"
]].copy()

df_postagem["tempo_postagem_dias"] = (
    df_postagem["order_delivered_carrier_date"] -
    df_postagem["order_approved_at"]
).dt.days

print(f"\nTempo de postagem (dias):")
print(df_postagem["tempo_postagem_dias"].describe().round(2))
print(f"Nulos:     {df_postagem['tempo_postagem_dias'].isna().sum():,}")
print(f"Negativos: {(df_postagem['tempo_postagem_dias'] < 0).sum():,}")

# ── 3. Associa variáveis à primeira compra de cada cliente elegível ──
# Usamos apenas a primeira compra para evitar data leakage
primeira_ordem_teste = (df_elegivel
    .sort_values("order_approved_at")
    .groupby("customer_unique_id")["order_id"]
    .first().reset_index()
    .rename(columns={"order_id": "order_id_primeira"}))

# Frete da primeira compra
primeira_ordem_teste = primeira_ordem_teste.merge(
    items_frete,
    left_on="order_id_primeira",
    right_on="order_id", how="left")

# Tempo de postagem da primeira compra
primeira_ordem_teste = primeira_ordem_teste.merge(
    df_postagem[["order_id", "tempo_postagem_dias"]],
    left_on="order_id_primeira",
    right_on="order_id", how="left")

# Target
df_base_check = pd.read_csv("df_base_features.csv")[
    ["customer_unique_id", "target"]]

primeira_ordem_teste = primeira_ordem_teste.merge(
    df_base_check, on="customer_unique_id", how="left")

# ── 4. Análise de poder preditivo por quartil ──
# Se houver gradiente claro na taxa de recompra entre quartis,
# a variável tem poder preditivo e deve entrar no modelo.
# Se a taxa for uniforme entre quartis, a variável é descartada.

# Frete
primeira_ordem_teste["quartil_frete"] = pd.qcut(
    primeira_ordem_teste["frete_total"].fillna(0),
    q=4,
    labels=["Q1 (menor)", "Q2", "Q3", "Q4 (maior)"])

taxa_frete = (primeira_ordem_teste
    .groupby("quartil_frete", observed=False)["target"]
    .agg(recompraram="sum", total="count", taxa_recompra="mean")
    .round(4))

print(f"\nTaxa de recompra por quartil de frete:")
print(taxa_frete.to_string())

# Tempo de postagem
primeira_ordem_teste["quartil_postagem"] = pd.qcut(
    primeira_ordem_teste["tempo_postagem_dias"].fillna(2),
    q=4,
    labels=["Q1 (mais rápido)", "Q2", "Q3", "Q4 (mais lento)"])

taxa_postagem = (primeira_ordem_teste
    .groupby("quartil_postagem", observed=False)["target"]
    .agg(recompraram="sum", total="count", taxa_recompra="mean")
    .round(4))

print(f"\nTaxa de recompra por quartil de tempo de postagem:")
print(taxa_postagem.to_string())

# ── 5. Conclusão do teste ──
variacao_frete    = taxa_frete["taxa_recompra"].max() - taxa_frete["taxa_recompra"].min()
variacao_postagem = taxa_postagem["taxa_recompra"].max() - taxa_postagem["taxa_recompra"].min()

print(f"\n── Conclusão do teste ──")
print(f"Variação taxa recompra entre quartis de frete:          {variacao_frete:.4f} ({variacao_frete*100:.2f} pp)")
print(f"Variação taxa recompra entre quartis de tempo postagem: {variacao_postagem:.4f} ({variacao_postagem*100:.2f} pp)")
print(f"\nCritério de inclusão: variação > 1,0 pp entre quartis extremos")
print(f"Frete:          {'INCLUIR' if variacao_frete > 0.01 else 'DESCARTAR — ausência de gradiente preditivo'}")
print(f"Tempo postagem: {'INCLUIR' if variacao_postagem > 0.01 else 'DESCARTAR — ausência de gradiente preditivo'}")
