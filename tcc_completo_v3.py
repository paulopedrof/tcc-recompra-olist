"""
================================================================================
TCC — MBA em Data Science e Analytics — USP/Esalq                (versão 3)
================================================================================
Título: Predição de recompra em e-commerce brasileiro com machine learning
        e análise de sentimento
Autor: Paulo Pedro Filho | Orientador: Prof. João Vitor Matos Gonçalves

O QUE MUDOU EM RELAÇÃO À VERSÃO ANTERIOR (tcc_completo_comentado.py)
  1. Target: recompra = novo pedido, em DIA DIFERENTE do primeiro, feito até
     180 dias após a primeira compra. (Antes: qualquer segundo pedido, sem
     limite de prazo, inclusive pedidos feitos no mesmo dia.)
  2. Todas as variáveis vêm do PRIMEIRO pedido, inclusive atraso e flag de
     atraso. (Antes elas eram calculadas sobre todos os pedidos do cliente,
     o que vazava informação sobre a recompra.)
  3. A recência saiu: ela mede tempo de exposição, não comportamento. Entrou o
     mês da primeira compra.
  4. Clientes que já tinham recomprado ANTES da entrega do primeiro pedido saem
     da amostra: a recompra já era conhecida no ponto de previsão (entrega do
     primeiro pedido).
  5. Sentimento: saída do BERTimbau ajustado (bertimbau_colab.py), com uma
     avaliação por pedido (antes havia linhas duplicadas).
  6. O Optuna usa validação cruzada só no TREINO, com semente fixa. O conjunto
     de teste é usado uma única vez, para o relatório final.
  7. Random Forest também otimizada. Regressão Logística com variáveis dummy,
     log do valor, UMA variável de sentimento (positivo menos negativo) e
     tabela de coeficientes (statsmodels), com teste por bloco (estado, mês).
  7b. Testes de proporção (com IC95%) para durabilidade, atraso e nota.
  8. Novos: ablação, intervalos de confiança (bootstrap), limiar de decisão
     escolhido no treino, lift, tabelas em CSV e figuras no padrão USP/Esalq.
  9. Figuras: eixo vertical da curva precisão-recall limitado a 0,10 e
     vírgula decimal nos eixos e legendas.

ARQUIVOS NECESSÁRIOS NA PASTA DO PROJETO
  olist_orders_dataset.csv, olist_customers_dataset.csv,
  olist_order_payments_dataset.csv, olist_order_items_dataset.csv,
  olist_products_dataset.csv, olist_product_category_name_translation.csv,
  bertimbau_sentimento_reviews.csv   (gerado pelo bertimbau_colab.py)

COMO RODAR
  Spyder -> F5. Tempo estimado: 40 a 60 minutos (a otimização é a parte lenta).
  Para um teste rápido (2 minutos), troque MODO_TESTE para True logo abaixo.
================================================================================
"""

import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
import optuna
import shap
import matplotlib
matplotlib.use("Agg")  # salva as figuras em arquivo, sem abrir janela
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_curve, roc_curve,
                             precision_score, recall_score, f1_score,
                             confusion_matrix)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    TEM_STATSMODELS = True
except ImportError:
    TEM_STATSMODELS = False

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÕES
# ══════════════════════════════════════════════════════════════════════════════
SEMENTE      = 42     # fixa o sorteio: mesmos resultados a cada execução
JANELA_DIAS  = 180    # horizonte de recompra
N_TRIALS     = 30     # tentativas do Optuna por modelo
CV_FOLDS     = 5      # partes da validação cruzada (dentro do treino)
N_BOOTSTRAP  = 1000   # reamostragens para os intervalos de confiança
MODO_TESTE   = False  # True = execução rápida só para conferir se roda

if MODO_TESTE:
    N_TRIALS, CV_FOLDS, N_BOOTSTRAP = 2, 3, 100

np.random.seed(SEMENTE)
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", message=".*feature names.*")
warnings.filterwarnings("ignore", category=FutureWarning)


def titulo(texto):
    print("\n" + "=" * 70)
    print(texto)
    print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# 1. CARREGAMENTO
# ══════════════════════════════════════════════════════════════════════════════
titulo("1. Carregando os dados")

colunas_data = ["order_purchase_timestamp", "order_approved_at",
                "order_delivered_carrier_date", "order_delivered_customer_date",
                "order_estimated_delivery_date"]

orders    = pd.read_csv("olist_orders_dataset.csv", parse_dates=colunas_data)
customers = pd.read_csv("olist_customers_dataset.csv")
payments  = pd.read_csv("olist_order_payments_dataset.csv")
items     = pd.read_csv("olist_order_items_dataset.csv")
products  = pd.read_csv("olist_products_dataset.csv")
category  = pd.read_csv("olist_product_category_name_translation.csv")
sent      = pd.read_csv("bertimbau_sentimento_reviews.csv",
                        parse_dates=["review_creation_date"], low_memory=False)

print(f"pedidos: {len(orders):,} | clientes: {len(customers):,} | "
      f"itens: {len(items):,} | avaliações com sentimento: {len(sent):,}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. PEDIDOS ENTREGUES E CLIENTE ÚNICO
# ══════════════════════════════════════════════════════════════════════════════
# Só pedidos entregues representam uma compra concluída. O identificador que
# acompanha o cliente ao longo do tempo é o customer_unique_id.
titulo("2. Pedidos entregues e identificador único do cliente")

orders = orders[orders["order_status"] == "delivered"].copy()
df = orders.merge(customers[["customer_id", "customer_unique_id", "customer_state"]],
                  on="customer_id", how="left")
assert df["customer_unique_id"].notna().all(), "pedido sem cliente"

data_limite = df["order_purchase_timestamp"].max()
print(f"Pedidos entregues: {len(df):,}")
print(f"Clientes únicos com pedido entregue: {df['customer_unique_id'].nunique():,}")
print(f"Período das compras: {df['order_purchase_timestamp'].min().date()} "
      f"a {data_limite.date()}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. PRIMEIRO PEDIDO, TARGET E AMOSTRA ELEGÍVEL
# ══════════════════════════════════════════════════════════════════════════════
# - primeiro pedido: o mais antigo de cada cliente
# - recompra: novo pedido em dia diferente do primeiro, até 180 dias depois
# - elegível: primeira compra com pelo menos 180 dias de observação pela frente
# - saem da amostra os clientes que já tinham recomprado antes de o primeiro
#   pedido ser entregue (no ponto de previsão a recompra já era conhecida)
titulo("3. Primeiro pedido, target e amostra elegível")

df = df.sort_values(["customer_unique_id", "order_purchase_timestamp", "order_id"]
                    ).reset_index(drop=True)
df["data_compra"] = df["order_purchase_timestamp"].dt.normalize()

primeira = df.drop_duplicates("customer_unique_id", keep="first").copy()
primeira["data_primeira"] = primeira["data_compra"]
# Entrega do primeiro pedido; se estiver vazia, usa-se o prazo prometido
primeira["entrega_primeira"] = primeira["order_delivered_customer_date"].fillna(
    primeira["order_estimated_delivery_date"])

df = df.merge(primeira[["customer_unique_id", "data_primeira"]],
              on="customer_unique_id", how="left")
df["dias_desde_primeira"] = (df["data_compra"] - df["data_primeira"]).dt.days

# Pedidos feitos no mesmo dia do primeiro (carrinho dividido) não são recompra
mesmo_dia = df[(df["dias_desde_primeira"] == 0)].groupby("customer_unique_id").size()
n_mesmo_dia = int((mesmo_dia > 1).sum())

recompras = df[df["dias_desde_primeira"] >= 1]
rec = (recompras.groupby("customer_unique_id")
       .agg(dias_ate_recompra=("dias_desde_primeira", "min"),
            ts_recompra=("order_purchase_timestamp", "min"))
       .reset_index())

base = primeira.merge(rec, on="customer_unique_id", how="left")

data_corte = data_limite.normalize() - pd.Timedelta(days=JANELA_DIAS)
base["elegivel_janela"] = base["data_primeira"] <= data_corte
base["recompra_antes_entrega"] = (base["ts_recompra"].notna() &
                                  (base["ts_recompra"] < base["entrega_primeira"]))

n_clientes = len(base)
n_elegiveis = int(base["elegivel_janela"].sum())
n_antes = int((base["elegivel_janela"] & base["recompra_antes_entrega"]).sum())

amostra = base[base["elegivel_janela"] & ~base["recompra_antes_entrega"]].copy()
amostra["target"] = (amostra["dias_ate_recompra"] <= JANELA_DIAS).astype(int)

print(f"Data de corte (primeira compra até): {data_corte.date()}")
print(f"Clientes únicos:                              {n_clientes:,}")
print(f"Elegíveis pela janela de {JANELA_DIAS} dias:               {n_elegiveis:,}")
print(f"  excluídos por falta de janela:              {n_clientes - n_elegiveis:,}")
print(f"  excluídos por recompra antes da entrega:    {n_antes:,}")
print(f"Amostra final:                                {len(amostra):,}")
print(f"Recompradores (target=1): {int(amostra['target'].sum()):,} "
      f"({amostra['target'].mean() * 100:.2f}%)")
print(f"Clientes com mais de um pedido no mesmo dia da 1ª compra: {n_mesmo_dia:,}")
mediana_dias = amostra.loc[amostra["target"] == 1, "dias_ate_recompra"].median()
print(f"Mediana de dias até a recompra (entre os recompradores): {mediana_dias:.0f}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. VARIÁVEIS DO PRIMEIRO PEDIDO
# ══════════════════════════════════════════════════════════════════════════════
titulo("4. Construindo as variáveis (apenas do primeiro pedido)")
n_amostra = len(amostra)

# 4.1 Valor pago
valor = (payments.groupby("order_id")["payment_value"].sum().reset_index()
         .rename(columns={"payment_value": "valor_primeira_compra"}))
amostra = amostra.merge(valor, on="order_id", how="left")
amostra["valor_primeira_compra"] = amostra["valor_primeira_compra"].fillna(0)

# 4.2 Logística: folga de entrega (dias de antecedência; negativo = atraso)
amostra["atraso_dias"] = (amostra["order_estimated_delivery_date"] -
                          amostra["order_delivered_customer_date"]).dt.days
n_sem_entrega = int(amostra["atraso_dias"].isna().sum())
amostra["atraso_dias"] = amostra["atraso_dias"].fillna(amostra["atraso_dias"].median())
amostra["flag_atraso"] = (amostra["atraso_dias"] < 0).astype(int)

# 4.3 Frete e tempo de postagem (só para o teste descritivo da seção 5)
frete = (items.groupby("order_id")["freight_value"].sum().reset_index()
         .rename(columns={"freight_value": "frete_primeira"}))
amostra = amostra.merge(frete, on="order_id", how="left")
amostra["postagem_dias"] = (amostra["order_delivered_carrier_date"] -
                            amostra["order_approved_at"]).dt.days

# 4.4 Estado (10 maiores + "OUTROS") e mês da primeira compra
top_estados = amostra["customer_state"].value_counts().head(10).index
amostra["estado_grp"] = np.where(amostra["customer_state"].isin(top_estados),
                                 amostra["customer_state"], "OUTROS")
amostra["estado_cod"] = amostra["estado_grp"].astype("category").cat.codes
amostra["mes_primeira_compra"] = amostra["order_purchase_timestamp"].dt.month

# 4.5 Durabilidade do produto (categoria do primeiro item do primeiro pedido)
NAO_DURAVEIS = [
    "health_beauty", "perfumery", "food_drink", "food", "drinks",
    "diapers_and_hygiene", "baby", "stationery", "party_supplies",
    "christmas_supplies", "flowers", "arts_and_craftmanship",
    "books_general_interest", "books_technical", "books_imported",
    "fashion_male_clothing", "fashio_female_clothing",
    "fashion_childrens_clothes", "fashion_underwear_beach", "fashion_sport",
    "fashion_shoes", "fashion_bags_accessories", "pet_shop", "toys",
    "sports_leisure", "la_cuisine"]
DURAVEIS = [
    "art", "bed_bath_table", "furniture_decor", "computers_accessories",
    "housewares", "watches_gifts", "telephony", "cool_stuff", "garden_tools",
    "auto", "electronics", "office_furniture", "consoles_games",
    "small_appliances", "home_appliances", "home_appliances_2",
    "tablets_printing_image", "audio", "fixed_telephony", "air_conditioning",
    "computers", "cine_photo", "music", "musical_instruments",
    "cds_dvds_musicals", "dvds_blu_ray", "luggage_accessories",
    "furniture_bedroom", "furniture_living_room",
    "furniture_mattress_and_upholstery",
    "kitchen_dining_laundry_garden_furniture", "home_construction",
    "home_confort", "home_comfort_2", "construction_tools_construction",
    "costruction_tools_garden", "costruction_tools_tools",
    "construction_tools_lights", "construction_tools_safety",
    "signaling_and_security", "agro_industry_and_commerce",
    "industry_commerce_and_business", "security_and_services",
    "small_appliances_home_oven_and_coffee", "market_place"]
mapa_duravel = {c: 0 for c in NAO_DURAVEIS}
mapa_duravel.update({c: 1 for c in DURAVEIS})

primeiro_item = (items.sort_values(["order_id", "order_item_id"])
                 .drop_duplicates("order_id", keep="first")[["order_id", "product_id"]]
                 .merge(products[["product_id", "product_category_name"]],
                        on="product_id", how="left")
                 .merge(category, on="product_category_name", how="left"))
amostra = amostra.merge(primeiro_item[["order_id", "product_category_name_english"]],
                        on="order_id", how="left")
cat_ing = amostra["product_category_name_english"]
n_sem_categoria = int(cat_ing.isna().sum())
n_fora_mapa = int((cat_ing.notna() & ~cat_ing.isin(mapa_duravel.keys())).sum())
amostra["produto_duravel"] = cat_ing.map(mapa_duravel).fillna(1).astype(int)

# 4.6 Avaliação da primeira compra (nota e sentimento do BERTimbau)
# Uma avaliação por pedido: fica a mais recente.
sent = (sent.sort_values("review_creation_date")
        .drop_duplicates("order_id", keep="last"))
amostra = amostra.merge(
    sent[["order_id", "review_score", "tem_texto", "prob_neg", "prob_pos"]],
    on="order_id", how="left")
amostra = amostra.rename(columns={"review_score": "nota_primeira",
                                  "prob_neg": "sent_neg", "prob_pos": "sent_pos"})
n_sem_avaliacao = int(amostra["nota_primeira"].isna().sum())
amostra["nota_primeira"] = amostra["nota_primeira"].fillna(5.0)
amostra["tem_texto"] = amostra["tem_texto"].fillna(0).astype(int)
amostra["sent_neg"] = amostra["sent_neg"].fillna(0.0)
amostra["sent_pos"] = amostra["sent_pos"].fillna(0.0)

# Verificações de integridade
assert len(amostra) == n_amostra, "algum merge duplicou linhas"
assert amostra["customer_unique_id"].is_unique, "cliente repetido"

print(f"Clientes sem data de entrega (atraso imputado pela mediana): {n_sem_entrega}")
print(f"Clientes sem categoria de produto (tratados como duráveis):   {n_sem_categoria}")
print(f"Categorias fora do mapa de durabilidade:                      {n_fora_mapa}")
print(f"Clientes sem avaliação da primeira compra (nota 5 imputada):  {n_sem_avaliacao}")
print(f"Avaliações da primeira compra com texto:                      "
      f"{int(amostra['tem_texto'].sum()):,} ({amostra['tem_texto'].mean() * 100:.1f}%)")
print(f"Linhas: {len(amostra):,} | clientes únicos: {amostra['customer_unique_id'].nunique():,}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. ANÁLISE DESCRITIVA: taxa de recompra por faixa de cada variável
# ══════════════════════════════════════════════════════════════════════════════
titulo("5. Análise descritiva (taxa de recompra por faixa)")


def taxa_por(serie, nome):
    t = (amostra.groupby(serie, observed=False)["target"]
         .agg(recompraram="sum", total="count", taxa="mean").round(4))
    print(f"\nTaxa de recompra por {nome}:")
    print(t.to_string())
    return t


taxa_por("nota_primeira", "nota da avaliação")
taxa_por("produto_duravel", "durabilidade (1 = durável)")

faixa_atraso = pd.cut(amostra["atraso_dias"], [-np.inf, -1, 4, 9, 14, np.inf],
                      labels=["Atrasada", "0 a 4 dias antes", "5 a 9 dias antes",
                              "10 a 14 dias antes", "15 ou mais dias antes"])
t_atraso = taxa_por(faixa_atraso, "faixa de entrega")

taxa_por(pd.qcut(amostra["valor_primeira_compra"], 4, duplicates="drop",
                 labels=["Q1 (menor)", "Q2", "Q3", "Q4 (maior)"]),
         "quartil de valor da primeira compra")

# Variáveis candidatas descartadas: frete e tempo de postagem
print("\n── Teste de variáveis candidatas (critério: variação > 1,0 p.p.) ──")
for col, nome in [("frete_primeira", "frete"), ("postagem_dias", "tempo de postagem")]:
    q = pd.qcut(amostra[col].fillna(amostra[col].median()), 4, duplicates="drop")
    tx = amostra.groupby(q, observed=False)["target"].mean()
    var = tx.max() - tx.min()
    decisao = "INCLUIR" if var > 0.01 else "DESCARTAR (sem gradiente preditivo)"
    print(f"{nome:18s} variação entre quartis: {var * 100:.2f} p.p. -> {decisao}")

# Testes de duas proporções (z, bilateral) com IC95% da diferença.
# As três comparações foram sugeridas pela análise descritiva acima, portanto
# são exploratórias; com correção de Bonferroni (3 testes) o nível é 0,0167.
print("\n── Testes de proporção (amostra completa) ──")


def comparar_proporcoes(mask_a, mask_b, nome_a, nome_b):
    ya, yb = amostra.loc[mask_a, "target"], amostra.loc[mask_b, "target"]
    na, nb, ka, kb = len(ya), len(yb), ya.sum(), yb.sum()
    pa, pb = ka / na, kb / nb
    pool = (ka + kb) / (na + nb)
    z = (pa - pb) / np.sqrt(pool * (1 - pool) * (1 / na + 1 / nb))
    p = 2 * norm.sf(abs(z))
    ep = np.sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb)
    d = pa - pb
    print(f"{nome_a} ({pa * 100:.2f}%, n={na:,}) x {nome_b} ({pb * 100:.2f}%, n={nb:,}): "
          f"diferença {d * 100:+.2f} p.p. (IC95% {(d - 1.96 * ep) * 100:+.2f} a "
          f"{(d + 1.96 * ep) * 100:+.2f}), z={z:.2f}, p={p:.4f}")


comparar_proporcoes(amostra["produto_duravel"] == 0, amostra["produto_duravel"] == 1,
                    "Não durável", "Durável")
comparar_proporcoes(amostra["flag_atraso"] == 0, amostra["flag_atraso"] == 1,
                    "Entrega no prazo", "Entrega atrasada")
comparar_proporcoes(amostra["nota_primeira"] >= 4, amostra["nota_primeira"] <= 2,
                    "Nota 4 ou 5", "Nota 1 ou 2")
print("Nível de significância com correção de Bonferroni (3 testes): 0,0167")

# ══════════════════════════════════════════════════════════════════════════════
# 6. MATRIZES DE VARIÁVEIS, DIVISÃO TREINO/TESTE
# ══════════════════════════════════════════════════════════════════════════════
titulo("6. Divisão treino/teste")

FEATURES_ARV = ["valor_primeira_compra", "atraso_dias", "flag_atraso",
                "estado_cod", "mes_primeira_compra", "produto_duravel",
                "nota_primeira", "tem_texto", "sent_neg", "sent_pos"]

NOMES_BONITOS = {
    "valor_primeira_compra": "Valor da primeira compra",
    "atraso_dias": "Antecedência da entrega (dias)",
    "flag_atraso": "Entrega atrasada (sim/não)",
    "estado_cod": "Estado do cliente",
    "mes_primeira_compra": "Mês da primeira compra",
    "produto_duravel": "Produto durável",
    "nota_primeira": "Nota da avaliação",
    "tem_texto": "Avaliação com texto",
    "sent_neg": "Sentimento negativo (BERTimbau)",
    "sent_pos": "Sentimento positivo (BERTimbau)",
}

y = amostra["target"]
X_arv = amostra[FEATURES_ARV].copy()

idx_treino, idx_teste = train_test_split(
    amostra.index, test_size=0.20, stratify=y, random_state=SEMENTE)

X_train_arv, X_test_arv = X_arv.loc[idx_treino], X_arv.loc[idx_teste]
y_train, y_test = y.loc[idx_treino], y.loc[idx_teste]
y_test_np = y_test.values

# Matriz para os modelos lineares: log do valor, atraso limitado aos
# percentis 1% e 99% do treino, estado e mês como variáveis dummy
lim_atraso = (X_train_arv["atraso_dias"].quantile(0.01),
              X_train_arv["atraso_dias"].quantile(0.99))
ref_estado = amostra["estado_grp"].value_counts().idxmax()

X_lin = pd.DataFrame(index=amostra.index)
X_lin["log_valor"] = np.log1p(amostra["valor_primeira_compra"])
X_lin["atraso_dias"] = amostra["atraso_dias"].clip(*lim_atraso)
X_lin["flag_atraso"] = amostra["flag_atraso"]
X_lin["produto_duravel"] = amostra["produto_duravel"]
X_lin["nota_primeira"] = amostra["nota_primeira"]
# Uma só variável de sentimento (positivo menos negativo; 0 sem texto): as três
# variáveis de texto separadas eram colineares por construção (VIF > 10)
X_lin["sent_liquido"] = amostra["sent_pos"] - amostra["sent_neg"]
dum_uf = pd.get_dummies(amostra["estado_grp"], prefix="uf").drop(columns=f"uf_{ref_estado}")
dum_mes = pd.get_dummies(amostra["mes_primeira_compra"], prefix="mes", drop_first=True)
X_lin = pd.concat([X_lin, dum_uf, dum_mes], axis=1).astype(float)
X_train_lin, X_test_lin = X_lin.loc[idx_treino], X_lin.loc[idx_teste]

print(f"Treino: {len(y_train):,} (recompradores: {y_train.mean() * 100:.2f}%)")
print(f"Teste:  {len(y_test):,} (recompradores: {y_test.mean() * 100:.2f}%)")
print(f"Estado de referência da regressão: {ref_estado}")

skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEMENTE)
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight (não recompradores / recompradores): {scale_pos:.2f}")


# ── Funções de apoio ──
def cv_ap(estimador, X, yy):
    """AUC-PR em cada parte da validação cruzada (só dentro do treino)."""
    notas = []
    for i_tr, i_va in skf.split(X, yy):
        m = clone(estimador).fit(X.iloc[i_tr], yy.iloc[i_tr])
        notas.append(average_precision_score(yy.iloc[i_va],
                                             m.predict_proba(X.iloc[i_va])[:, 1]))
    return np.array(notas)


def cv_oof(estimador, X, yy):
    """Probabilidades 'fora da amostra' de cada cliente do treino."""
    oof = np.zeros(len(yy))
    for i_tr, i_va in skf.split(X, yy):
        m = clone(estimador).fit(X.iloc[i_tr], yy.iloc[i_tr])
        oof[i_va] = m.predict_proba(X.iloc[i_va])[:, 1]
    return oof


def limiar_f1(y_true, prob):
    """Limiar de decisão que maximiza o F1 (escolhido apenas no treino)."""
    p, r, thr = precision_recall_curve(y_true, prob)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return thr[int(np.argmax(f1))]


def lift_topo(y_true, prob, frac=0.10):
    """Entre os 10% de clientes com maior probabilidade: precisão, captura e lift."""
    n = int(np.ceil(frac * len(y_true)))
    top = np.argsort(-prob)[:n]
    prec = y_true[top].mean()
    return prec, y_true[top].sum() / y_true.sum(), prec / y_true.mean()


# ══════════════════════════════════════════════════════════════════════════════
# 7. MODELOS BASE (parametrização inicial, sem otimização)
# ══════════════════════════════════════════════════════════════════════════════
titulo("7. Modelos com parametrização base")

lr_base = make_pipeline(StandardScaler(),
                        LogisticRegression(class_weight="balanced", max_iter=2000,
                                           random_state=SEMENTE))
rf_base = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                 n_jobs=-1, random_state=SEMENTE)
xgb_base = XGBClassifier(n_estimators=300, scale_pos_weight=scale_pos,
                         eval_metric="aucpr", random_state=SEMENTE, verbosity=0)
lgbm_base = LGBMClassifier(n_estimators=300, scale_pos_weight=scale_pos,
                           random_state=SEMENTE, verbosity=-1)

p_teste = {}  # probabilidades no teste, de todos os modelos
for nome, est, Xtr, Xte in [
        ("Regressão Logística (base)", lr_base, X_train_lin, X_test_lin),
        ("Random Forest (base)", rf_base, X_train_arv, X_test_arv),
        ("XGBoost (base)", xgb_base, X_train_arv, X_test_arv),
        ("LightGBM (base)", lgbm_base, X_train_arv, X_test_arv)]:
    est.fit(Xtr, y_train)
    p_teste[nome] = est.predict_proba(Xte)[:, 1]
    print(f"{nome:28s} AUC-ROC {roc_auc_score(y_test_np, p_teste[nome]):.4f} | "
          f"AUC-PR {average_precision_score(y_test_np, p_teste[nome]):.4f}")
print(f"Referência aleatória do AUC-PR (prevalência no teste): {y_test.mean():.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 8. REGRESSÃO LOGÍSTICA: COEFICIENTES, ODDS RATIOS E VIF
# ══════════════════════════════════════════════════════════════════════════════
titulo("8. Regressão logística (inferência, statsmodels)")

if TEM_STATSMODELS:
    try:
        Xc = sm.add_constant(X_train_lin, has_constant="add")
        logit = sm.Logit(y_train, Xc).fit(disp=0, maxiter=300)
        ic = logit.conf_int()
        tab_logit = pd.DataFrame({
            "coeficiente": logit.params,
            "odds_ratio": np.exp(logit.params),
            "or_ic95_inf": np.exp(ic[0]),
            "or_ic95_sup": np.exp(ic[1]),
            "p_valor": logit.pvalues})
        tab_logit["VIF"] = [variance_inflation_factor(Xc.values, i)
                            for i in range(Xc.shape[1])]
        tab_logit.loc["const", "VIF"] = np.nan
        print(tab_logit.round(4).to_string())
        print(f"\nPseudo-R² de McFadden: {logit.prsquared:.4f}")
        p_logit = logit.predict(sm.add_constant(X_test_lin, has_constant="add"))
        print(f"Teste (statsmodels): AUC-ROC {roc_auc_score(y_test_np, p_logit):.4f} | "
              f"AUC-PR {average_precision_score(y_test_np, p_logit):.4f}")
        tab_logit.round(4).to_csv("tabela_logit_v3.csv")
        # Teste da razão de verossimilhança: o bloco todo acrescenta algo?
        for nome_bloco, prefixo in [("Estado (10 variáveis dummy)", "uf_"),
                                    ("Mês da compra (11 variáveis dummy)", "mes_")]:
            cols_bloco = [c for c in X_train_lin.columns if c.startswith(prefixo)]
            reduzido = sm.Logit(y_train, sm.add_constant(
                X_train_lin.drop(columns=cols_bloco), has_constant="add")
            ).fit(disp=0, maxiter=300)
            est_lr = 2 * (logit.llf - reduzido.llf)
            print(f"Teste de bloco, {nome_bloco}: qui-quadrado {est_lr:.2f} "
                  f"({len(cols_bloco)} g.l.), p = {chi2.sf(est_lr, len(cols_bloco)):.4f}")
    except Exception as erro:
        print(f"Não foi possível estimar o Logit: {erro}")
else:
    print("statsmodels não instalado. Instale com: pip install statsmodels")

# ══════════════════════════════════════════════════════════════════════════════
# 9. OTIMIZAÇÃO DE HIPERPARÂMETROS (Optuna, validação cruzada no treino)
# ══════════════════════════════════════════════════════════════════════════════
# O Optuna usa otimização bayesiana: aprende com cada tentativa onde procurar
# a seguir. A nota de cada tentativa é o AUC-PR médio na validação cruzada
# DENTRO DO TREINO. O conjunto de teste não participa.
titulo(f"9. Otimização com Optuna ({N_TRIALS} tentativas, {CV_FOLDS} partes)")


def fab_rf(t):
    return RandomForestClassifier(
        n_estimators=t.suggest_int("n_estimators", 200, 500),
        max_depth=t.suggest_int("max_depth", 4, 20),
        min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 40, log=True),
        max_features=t.suggest_float("max_features", 0.3, 1.0),
        class_weight="balanced_subsample", n_jobs=-1, random_state=SEMENTE)


def fab_xgb(t):
    return XGBClassifier(
        n_estimators=t.suggest_int("n_estimators", 100, 800),
        max_depth=t.suggest_int("max_depth", 3, 9),
        learning_rate=t.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample=t.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=t.suggest_float("colsample_bytree", 0.6, 1.0),
        min_child_weight=t.suggest_int("min_child_weight", 1, 20),
        reg_lambda=t.suggest_float("reg_lambda", 0.01, 10.0, log=True),
        scale_pos_weight=scale_pos, eval_metric="aucpr",
        random_state=SEMENTE, verbosity=0)


def fab_lgbm(t):
    # subsample_freq=1 é necessário para o subsample ter efeito no LightGBM
    return LGBMClassifier(
        n_estimators=t.suggest_int("n_estimators", 100, 800),
        max_depth=t.suggest_int("max_depth", 3, 10),
        num_leaves=t.suggest_int("num_leaves", 15, 127),
        learning_rate=t.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample=t.suggest_float("subsample", 0.6, 1.0), subsample_freq=1,
        colsample_bytree=t.suggest_float("colsample_bytree", 0.6, 1.0),
        min_child_samples=t.suggest_int("min_child_samples", 10, 100),
        reg_lambda=t.suggest_float("reg_lambda", 0.01, 10.0, log=True),
        scale_pos_weight=scale_pos, random_state=SEMENTE, verbosity=-1)


def otimizar(nome, fabrica):
    def objetivo(trial):
        return cv_ap(fabrica(trial), X_train_arv, y_train).mean()
    estudo = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEMENTE))
    estudo.optimize(objetivo, n_trials=N_TRIALS)
    print(f"{nome:10s} melhor AUC-PR na validação cruzada: {estudo.best_value:.4f}")
    return estudo


estudos = {"Random Forest": (otimizar("Random Forest", fab_rf), fab_rf),
           "XGBoost": (otimizar("XGBoost", fab_xgb), fab_xgb),
           "LightGBM": (otimizar("LightGBM", fab_lgbm), fab_lgbm)}

# Modelos otimizados (reconstruídos com os melhores parâmetros)
modelos_otim = {nome: fab(optuna.trial.FixedTrial(est.best_params))
                for nome, (est, fab) in estudos.items()}

# Tabela de hiperparâmetros: espaço de busca e valor selecionado
linhas_hp = []
for nome, (est, fab) in estudos.items():
    distrib = est.best_trial.distributions
    for par, valor_sel in est.best_params.items():
        d = distrib[par]
        espaco = f"{d.low} a {d.high}" + (" (escala log)" if d.log else "")
        v = round(valor_sel, 4) if isinstance(valor_sel, float) else valor_sel
        linhas_hp.append({"Modelo": nome, "Hiperparâmetro": par,
                          "Espaço de busca": espaco, "Valor selecionado": v})
tab_hp = pd.DataFrame(linhas_hp)
print("\nHiperparâmetros selecionados:")
print(tab_hp.to_string(index=False))
tab_hp.to_csv("tabela_hiperparametros_v3.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════════
# 10. AVALIAÇÃO NO TESTE (uma única vez) COM LIMIAR ESCOLHIDO NO TREINO
# ══════════════════════════════════════════════════════════════════════════════
titulo("10. Avaliação final no conjunto de teste")

finais = {"Regressão Logística": (lr_base, X_train_lin, X_test_lin),
          "Random Forest (otimizada)": (modelos_otim["Random Forest"], X_train_arv, X_test_arv),
          "XGBoost (otimizado)": (modelos_otim["XGBoost"], X_train_arv, X_test_arv),
          "LightGBM (otimizado)": (modelos_otim["LightGBM"], X_train_arv, X_test_arv)}

linhas, ap_cv, limiares, ajustados, oof_ap = [], {}, {}, {}, {}
for nome, (est, Xtr, Xte) in finais.items():
    est = clone(est).fit(Xtr, y_train)
    ajustados[nome] = est
    prob = est.predict_proba(Xte)[:, 1]
    p_teste[nome] = prob
    oof = cv_oof(est, Xtr, y_train)
    oof_ap[nome] = average_precision_score(y_train, oof)
    lim = limiar_f1(y_train.values, oof)
    limiares[nome] = lim
    pred = (prob >= lim).astype(int)
    prec_top, capt_top, lift = lift_topo(y_test_np, prob)
    linhas.append({
        "Modelo": nome,
        "AUC-PR (validação cruzada)": round(oof_ap[nome], 4),
        "AUC-ROC (teste)": round(roc_auc_score(y_test_np, prob), 4),
        "AUC-PR (teste)": round(average_precision_score(y_test_np, prob), 4),
        "Limiar": round(float(lim), 3),
        "Precisão": round(precision_score(y_test_np, pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test_np, pred, zero_division=0), 4),
        "F1": round(f1_score(y_test_np, pred, zero_division=0), 4),
        "Captura no 1º decil": round(capt_top, 4),
        "Lift no 1º decil": round(lift, 2)})

# Intervalos de confiança por bootstrap (mesmas reamostragens para todos)
rng = np.random.default_rng(SEMENTE)
n_teste = len(y_test_np)
indices_boot = [rng.integers(0, n_teste, n_teste) for _ in range(N_BOOTSTRAP)]


def ic_boot(prob):
    roc, ap = [], []
    for idx in indices_boot:
        yt = y_test_np[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        roc.append(roc_auc_score(yt, prob[idx]))
        ap.append(average_precision_score(yt, prob[idx]))
    return np.percentile(roc, [2.5, 97.5]), np.percentile(ap, [2.5, 97.5])


for linha in linhas:
    r, a = ic_boot(p_teste[linha["Modelo"]])
    linha["IC95% AUC-ROC"] = f"{r[0]:.3f} a {r[1]:.3f}"
    linha["IC95% AUC-PR"] = f"{a[0]:.3f} a {a[1]:.3f}"

tab_met = pd.DataFrame(linhas)
print(tab_met.T.to_string(header=False))
tab_met.to_csv("tabela_metricas_v3.csv", index=False)

# Modelos base e otimizados lado a lado (AUC no teste)
print("\nComparação base x otimizado (AUC-PR no teste):")
for base_n, otim_n in [("Random Forest (base)", "Random Forest (otimizada)"),
                       ("XGBoost (base)", "XGBoost (otimizado)"),
                       ("LightGBM (base)", "LightGBM (otimizado)")]:
    print(f"{base_n:24s} {average_precision_score(y_test_np, p_teste[base_n]):.4f}  ->  "
          f"{otim_n:26s} {average_precision_score(y_test_np, p_teste[otim_n]):.4f}")

# Diferença pareada entre os dois modelos de boosting
dif = []
for idx in indices_boot:
    yt = y_test_np[idx]
    if yt.sum() == 0 or yt.sum() == len(yt):
        continue
    dif.append(average_precision_score(yt, p_teste["XGBoost (otimizado)"][idx]) -
               average_precision_score(yt, p_teste["LightGBM (otimizado)"][idx]))
print(f"\nDiferença de AUC-PR XGBoost - LightGBM: média {np.mean(dif):+.4f}, "
      f"IC95% {np.percentile(dif, 2.5):+.4f} a {np.percentile(dif, 97.5):+.4f}")

# Modelo finalista: melhor AUC-PR na validação cruzada (não no teste)
arvores = ["Random Forest (otimizada)", "XGBoost (otimizado)", "LightGBM (otimizado)"]
finalista = max(arvores, key=lambda n: oof_ap[n])
modelo_final = ajustados[finalista]
prob_final = p_teste[finalista]
print(f"\nFinalista (maior AUC-PR na validação cruzada): {finalista}")

pred_final = (prob_final >= limiares[finalista]).astype(int)
vn, fp, fn, vp = confusion_matrix(y_test_np, pred_final).ravel()
print(f"\nMatriz de confusão ({finalista}, limiar {limiares[finalista]:.3f}):")
print(f"                 previsto 0   previsto 1")
print(f"  real 0         {vn:>10,}   {fp:>10,}")
print(f"  real 1         {fn:>10,}   {vp:>10,}")

# ══════════════════════════════════════════════════════════════════════════════
# 11. ABLAÇÃO: o que cada grupo de variáveis acrescenta
# ══════════════════════════════════════════════════════════════════════════════
# Mesmo modelo, mesmos parâmetros, mesmas partes da validação cruzada (treino).
# Cada linha retira um grupo de variáveis e compara com o modelo completo.
titulo("11. Ablação (AUC-PR na validação cruzada do treino)")

GRUPOS = {
    "Valor da compra": ["valor_primeira_compra"],
    "Logística (entrega)": ["atraso_dias", "flag_atraso"],
    "Estado": ["estado_cod"],
    "Mês da compra": ["mes_primeira_compra"],
    "Durabilidade do produto": ["produto_duravel"],
    "Nota da avaliação": ["nota_primeira"],
    "Texto (BERTimbau)": ["tem_texto", "sent_neg", "sent_pos"],
}
variantes = {"Modelo completo": FEATURES_ARV}
for g, cols in GRUPOS.items():
    variantes[f"Sem {g.lower()}"] = [c for c in FEATURES_ARV if c not in cols]
variantes["Sem avaliação (nota e texto)"] = [
    c for c in FEATURES_ARV if c not in GRUPOS["Nota da avaliação"] + GRUPOS["Texto (BERTimbau)"]]

est_abl = clone(modelo_final)
resultados_abl = {nome: cv_ap(est_abl, X_train_arv[cols], y_train)
                  for nome, cols in variantes.items()}
completo = resultados_abl["Modelo completo"]
linhas_abl = []
for nome, notas in resultados_abl.items():
    d = notas - completo
    linhas_abl.append({"Variante": nome,
                       "AUC-PR médio": round(notas.mean(), 4),
                       "Desvio entre partes": round(notas.std(ddof=1), 4),
                       "Diferença para o completo": round(d.mean(), 4)})
tab_abl = pd.DataFrame(linhas_abl)
print(f"Modelo usado na ablação: {finalista}")
print(tab_abl.to_string(index=False))
tab_abl.to_csv("tabela_ablacao_v3.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════════
# 12. INTERPRETABILIDADE (SHAP) NO MODELO FINALISTA
# ══════════════════════════════════════════════════════════════════════════════
titulo("12. SHAP no modelo finalista")

amostra_shap = X_test_arv.sample(n=min(3000, len(X_test_arv)), random_state=SEMENTE)
sv = shap.TreeExplainer(modelo_final).shap_values(amostra_shap)
if isinstance(sv, list):
    sv = sv[1]
elif getattr(sv, "ndim", 2) == 3:
    sv = sv[:, :, 1]

unidade = "probabilidade" if "Random Forest" in finalista else "log-odds"
imp = pd.DataFrame({
    "variavel": amostra_shap.columns,
    "shap_medio_abs": np.abs(sv).mean(axis=0),
    # sinal: correlação (Spearman) entre o valor da variável e o seu SHAP
    "correlacao_valor_shap": [pd.Series(sv[:, j]).corr(
        amostra_shap.iloc[:, j].reset_index(drop=True), method="spearman")
        for j in range(sv.shape[1])]
}).sort_values("shap_medio_abs", ascending=False).reset_index(drop=True)
imp["variavel_legivel"] = imp["variavel"].map(NOMES_BONITOS)
# O sinal não faz sentido para um código de estado nem para variáveis de texto
# (que valem zero quando não há texto)
imp.loc[imp["variavel"].isin(["estado_cod", "tem_texto", "sent_neg", "sent_pos"]),
        "correlacao_valor_shap"] = np.nan
print(f"SHAP médio absoluto (unidade: {unidade}; amostra de {len(amostra_shap):,} clientes do teste)")
print(imp[["variavel", "shap_medio_abs", "correlacao_valor_shap"]].round(4).to_string(index=False))
imp.round(4).to_csv("shap_importancia_v3.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════════
# 13. FIGURAS (padrão USP/Esalq: Arial, sem grade, sem moldura, eixos pretos)
#     O título da figura vai no Word, abaixo da imagem.
# ══════════════════════════════════════════════════════════════════════════════
titulo("13. Gerando as figuras")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11, "axes.linewidth": 1.5, "axes.edgecolor": "black",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
    "xtick.color": "black", "ytick.color": "black", "axes.labelcolor": "black"})

from matplotlib.ticker import FuncFormatter


def virgula(casas):
    """Números dos eixos com vírgula decimal."""
    return FuncFormatter(lambda v, _: f"{v:.{casas}f}".replace(".", ","))


def n3(x):
    """Número com 3 casas e vírgula decimal (legendas)."""
    return f"{x:.3f}".replace(".", ",")


curvas = {n: p_teste[n] for n in finais}


def salvar(nome_arquivo):
    plt.tight_layout()
    plt.savefig(nome_arquivo, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figura salva: {nome_arquivo}")


# Figura 1 do arquivo: curva ROC (Figura 2 no TCC)
fig, ax = plt.subplots(figsize=(6.3, 4.2))
for nome, p in curvas.items():
    fpr, tpr, _ = roc_curve(y_test_np, p)
    ax.plot(fpr, tpr, linewidth=1.6, label=f"{nome} ({n3(roc_auc_score(y_test_np, p))})")
ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=0.8)
ax.xaxis.set_major_formatter(virgula(1))
ax.yaxis.set_major_formatter(virgula(1))
ax.set_xlabel("Taxa de falsos positivos")
ax.set_ylabel("Taxa de verdadeiros positivos")
ax.legend(loc="lower right", fontsize=9)
salvar("figura1_curva_roc.png")

# Figura 2 do arquivo: precisão-recall (Figura 3 no TCC)
# O eixo vertical é limitado a 0,10: o pico de precisão perto de recall zero
# (poucos casos) esticaria o eixo até 1,0 e esconderia as curvas. Avisar na
# legenda da figura no Word.
fig, ax = plt.subplots(figsize=(6.3, 4.2))
for nome, p in curvas.items():
    pr, rc, _ = precision_recall_curve(y_test_np, p)
    ax.plot(rc, pr, linewidth=1.6,
            label=f"{nome} ({n3(average_precision_score(y_test_np, p))})")
ax.axhline(y_test.mean(), color="black", linestyle="--", linewidth=0.8,
           label=f"Referência aleatória ({n3(y_test.mean())})")
ax.set_ylim(0, 0.10)
ax.xaxis.set_major_formatter(virgula(1))
ax.yaxis.set_major_formatter(virgula(2))
ax.set_xlabel("Recall")
ax.set_ylabel("Precisão")
ax.legend(loc="upper right", fontsize=9)
salvar("figura2_curva_precisao_recall.png")

# Figura 3 do arquivo: importância SHAP (Figura 4 no TCC)
fig, ax = plt.subplots(figsize=(6.3, 4.2))
ordem = imp.sort_values("shap_medio_abs")
ax.barh(ordem["variavel_legivel"], ordem["shap_medio_abs"], color="#4C72B0")
ax.xaxis.set_major_formatter(virgula(2))
ax.set_xlabel(f"SHAP médio absoluto ({unidade})")
salvar("figura3_importancia_shap.png")

# Figura 4 do arquivo: taxa de recompra por faixa de entrega (Figura 1 no TCC)
fig, ax = plt.subplots(figsize=(6.3, 4.2))
ax.bar(range(len(t_atraso)), t_atraso["taxa"] * 100, color="#4C72B0")
ax.set_xticks(range(len(t_atraso)))
ax.set_xticklabels([str(i) for i in t_atraso.index], rotation=20, ha="right")
ax.yaxis.set_major_formatter(virgula(2))
ax.set_ylabel("Taxa de recompra em 180 dias (%)")
ax.set_xlabel("Entrega da primeira compra em relação ao prazo prometido")
salvar("figura4_recompra_por_entrega.png")

# ══════════════════════════════════════════════════════════════════════════════
# 14. CHECKPOINT E RESUMO
# ══════════════════════════════════════════════════════════════════════════════
amostra[["customer_unique_id", "target"] + FEATURES_ARV].to_csv(
    "base_modelagem_v3.csv", index=False)

titulo("CONCLUÍDO")
print(f"Amostra: {len(amostra):,} clientes | recompradores: {amostra['target'].mean() * 100:.2f}%")
print(f"Modelo finalista: {finalista}")
print("Arquivos gerados: tabela_metricas_v3.csv, tabela_hiperparametros_v3.csv,")
print("  tabela_ablacao_v3.csv, tabela_logit_v3.csv, shap_importancia_v3.csv,")
print("  base_modelagem_v3.csv, figura1..figura4 (.png)")
