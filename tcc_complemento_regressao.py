"""
================================================================================
TCC — COMPLEMENTO: regressão logística corrigida e testes de proporção
================================================================================
Roda em 1 a 2 minutos a partir do arquivo base_modelagem_v3.csv, gerado pelo
tcc_completo_v3.py. NÃO repete a otimização dos modelos de árvores: esses
resultados não mudam (semente fixa).

O que este script faz
  0. Confere se reproduz a regressão da execução anterior (se não reproduzir,
     para: a divisão treino/teste ou as variáveis estariam diferentes).
  1. Refaz a regressão com UMA variável de sentimento (positivo menos negativo),
     porque tem_texto, sent_neg e sent_pos eram colineares (VIF > 10).
  2. Teste de bloco (estado, mês), odds ratios e VIF.
  3. Métricas da regressão no teste (mesmo formato da tabela dos outros modelos).
  4. Testes de proporção (durabilidade, atraso, nota) com IC95%.

Como rodar: Spyder -> F5, com a pasta do projeto como diretório de trabalho.
================================================================================
"""

import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_curve, precision_score,
                             recall_score, f1_score)

warnings.filterwarnings("ignore", category=FutureWarning)

SEMENTE = 42          # a mesma do script principal
CV_FOLDS = 5          # a mesma do script principal
N_BOOTSTRAP = 1000    # o mesmo do script principal
VERIFICAR_REPRODUCAO = True

# Valores da regressão ANTERIOR (execução de 05/10), para conferir a reprodução
ESPERADO_ANTIGO = {"const": -3.9212, "log_valor": -0.0289, "uf_BA": -0.4160,
                   "uf_OUTROS": -0.5968, "uf_SC": -0.5635}

# O script principal guarda o estado como código (ordem alfabética dos grupos)
NOMES_UF = ["BA", "DF", "ES", "GO", "MG", "OUTROS", "PR", "RJ", "RS", "SC", "SP"]


def titulo(texto):
    print("\n" + "=" * 70)
    print(texto)
    print("=" * 70)


# ── Dados e divisão treino/teste (idêntica à do script principal) ──
dados = pd.read_csv("base_modelagem_v3.csv")
y = dados["target"]
dados["estado_grp"] = dados["estado_cod"].map(dict(enumerate(NOMES_UF)))
assert dados["estado_grp"].notna().all(), "código de estado inesperado"

idx_treino, idx_teste = train_test_split(
    dados.index, test_size=0.20, stratify=y, random_state=SEMENTE)
y_train, y_test = y.loc[idx_treino], y.loc[idx_teste]
y_test_np = y_test.values
print(f"Amostra: {len(dados):,} | treino: {len(y_train):,} | teste: {len(y_test):,} | "
      f"recompradores: {y.mean() * 100:.2f}%")

lim_atraso = (dados.loc[idx_treino, "atraso_dias"].quantile(0.01),
              dados.loc[idx_treino, "atraso_dias"].quantile(0.99))
ref_estado = dados["estado_grp"].value_counts().idxmax()


def montar_linear(especificacao):
    """Matriz da regressão. 'antiga' = 3 variáveis de texto; 'nova' = sentimento líquido."""
    X = pd.DataFrame(index=dados.index)
    X["log_valor"] = np.log1p(dados["valor_primeira_compra"])
    X["atraso_dias"] = dados["atraso_dias"].clip(*lim_atraso)
    X["flag_atraso"] = dados["flag_atraso"]
    X["produto_duravel"] = dados["produto_duravel"]
    X["nota_primeira"] = dados["nota_primeira"]
    if especificacao == "antiga":
        X["tem_texto"] = dados["tem_texto"]
        X["sent_neg"] = dados["sent_neg"]
        X["sent_pos"] = dados["sent_pos"]
    else:
        X["sent_liquido"] = dados["sent_pos"] - dados["sent_neg"]
    dum_uf = pd.get_dummies(dados["estado_grp"], prefix="uf").drop(columns=f"uf_{ref_estado}")
    dum_mes = pd.get_dummies(dados["mes_primeira_compra"], prefix="mes", drop_first=True)
    return pd.concat([X, dum_uf, dum_mes], axis=1).astype(float)


# ══════════════════════════════════════════════════════════════════════════════
# 0. CONFERÊNCIA: reproduz a regressão anterior?
# ══════════════════════════════════════════════════════════════════════════════
titulo("0. Conferência com a execução anterior")
X_antiga = montar_linear("antiga")
logit_antigo = sm.Logit(y_train, sm.add_constant(X_antiga.loc[idx_treino], has_constant="add")
                        ).fit(disp=0, maxiter=300)
ok = True
for nome, valor in ESPERADO_ANTIGO.items():
    obtido = logit_antigo.params[nome]
    bate = abs(obtido - valor) < 0.0006
    ok = ok and bate
    print(f"{nome:10s} esperado {valor:+.4f} | obtido {obtido:+.4f} | {'OK' if bate else 'DIFERENTE'}")
if VERIFICAR_REPRODUCAO and not ok:
    raise SystemExit("A conferência falhou: não rode o restante. Me mande este output.")
print("Conferência:", "reproduziu a execução anterior." if ok else "ignorada (VERIFICAR_REPRODUCAO=False).")

# ══════════════════════════════════════════════════════════════════════════════
# 1. REGRESSÃO LOGÍSTICA CORRIGIDA (inferência)
# ══════════════════════════════════════════════════════════════════════════════
titulo("1. Regressão logística corrigida (statsmodels)")
X_lin = montar_linear("nova")
X_train_lin, X_test_lin = X_lin.loc[idx_treino], X_lin.loc[idx_teste]

Xc = sm.add_constant(X_train_lin, has_constant="add")
logit = sm.Logit(y_train, Xc).fit(disp=0, maxiter=300)
ic = logit.conf_int()
tab_logit = pd.DataFrame({
    "coeficiente": logit.params,
    "odds_ratio": np.exp(logit.params),
    "or_ic95_inf": np.exp(ic[0]),
    "or_ic95_sup": np.exp(ic[1]),
    "p_valor": logit.pvalues})
tab_logit["VIF"] = [variance_inflation_factor(Xc.values, i) for i in range(Xc.shape[1])]
tab_logit.loc["const", "VIF"] = np.nan
print(tab_logit.round(4).to_string())
print(f"\nPseudo-R² de McFadden: {logit.prsquared:.4f}")
print(f"Maior VIF (sem a constante): {tab_logit['VIF'].max():.2f}")
p_logit = logit.predict(sm.add_constant(X_test_lin, has_constant="add"))
print(f"Teste (statsmodels): AUC-ROC {roc_auc_score(y_test_np, p_logit):.4f} | "
      f"AUC-PR {average_precision_score(y_test_np, p_logit):.4f}")
tab_logit.round(4).to_csv("tabela_logit_corrigida_v3.csv")

for nome_bloco, prefixo in [("Estado (10 variáveis dummy)", "uf_"),
                            ("Mês da compra (11 variáveis dummy)", "mes_")]:
    cols_bloco = [c for c in X_train_lin.columns if c.startswith(prefixo)]
    reduzido = sm.Logit(y_train, sm.add_constant(
        X_train_lin.drop(columns=cols_bloco), has_constant="add")).fit(disp=0, maxiter=300)
    est_lr = 2 * (logit.llf - reduzido.llf)
    print(f"Teste de bloco, {nome_bloco}: qui-quadrado {est_lr:.2f} "
          f"({len(cols_bloco)} g.l.), p = {chi2.sf(est_lr, len(cols_bloco)):.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. MÉTRICAS DA REGRESSÃO NO MESMO FORMATO DOS OUTROS MODELOS
# ══════════════════════════════════════════════════════════════════════════════
titulo("2. Regressão logística no conjunto de teste")
skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEMENTE)


def cv_oof(estimador, X, yy):
    oof = np.zeros(len(yy))
    for i_tr, i_va in skf.split(X, yy):
        m = clone(estimador).fit(X.iloc[i_tr], yy.iloc[i_tr])
        oof[i_va] = m.predict_proba(X.iloc[i_va])[:, 1]
    return oof


def limiar_f1(y_true, prob):
    p, r, thr = precision_recall_curve(y_true, prob)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return thr[int(np.argmax(f1))]


def lift_topo(y_true, prob, frac=0.10):
    n = int(np.ceil(frac * len(y_true)))
    top = np.argsort(-prob)[:n]
    prec = y_true[top].mean()
    return prec, y_true[top].sum() / y_true.sum(), prec / y_true.mean()


lr = make_pipeline(StandardScaler(),
                   LogisticRegression(class_weight="balanced", max_iter=2000,
                                      random_state=SEMENTE))
lr.fit(X_train_lin, y_train)
prob = lr.predict_proba(X_test_lin)[:, 1]
oof = cv_oof(lr, X_train_lin, y_train)
lim = limiar_f1(y_train.values, oof)
pred = (prob >= lim).astype(int)
prec_top, capt_top, lift = lift_topo(y_test_np, prob)

rng = np.random.default_rng(SEMENTE)
n_teste = len(y_test_np)
indices_boot = [rng.integers(0, n_teste, n_teste) for _ in range(N_BOOTSTRAP)]
roc_b, ap_b = [], []
for idx in indices_boot:
    yt = y_test_np[idx]
    if yt.sum() == 0 or yt.sum() == len(yt):
        continue
    roc_b.append(roc_auc_score(yt, prob[idx]))
    ap_b.append(average_precision_score(yt, prob[idx]))
r = np.percentile(roc_b, [2.5, 97.5])
a = np.percentile(ap_b, [2.5, 97.5])

linha = {
    "Modelo": "Regressão Logística",
    "AUC-PR (validação cruzada)": round(average_precision_score(y_train, oof), 4),
    "AUC-ROC (teste)": round(roc_auc_score(y_test_np, prob), 4),
    "AUC-PR (teste)": round(average_precision_score(y_test_np, prob), 4),
    "Limiar": round(float(lim), 3),
    "Precisão": round(precision_score(y_test_np, pred, zero_division=0), 4),
    "Recall": round(recall_score(y_test_np, pred, zero_division=0), 4),
    "F1": round(f1_score(y_test_np, pred, zero_division=0), 4),
    "Captura no 1º decil": round(capt_top, 4),
    "Lift no 1º decil": round(lift, 2),
    "IC95% AUC-ROC": f"{r[0]:.3f} a {r[1]:.3f}",
    "IC95% AUC-PR": f"{a[0]:.3f} a {a[1]:.3f}"}
for k, v in linha.items():
    print(f"{k:28s} {v}")
print(f"Referência aleatória do AUC-PR (prevalência no teste): {y_test.mean():.4f}")
pd.DataFrame([linha]).to_csv("linha_regressao_logistica_v3.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════════
# 3. TESTES DE PROPORÇÃO (amostra completa)
# ══════════════════════════════════════════════════════════════════════════════
titulo("3. Testes de proporção (amostra completa)")
print("As três comparações foram sugeridas pela análise descritiva, portanto são")
print("exploratórias. Com correção de Bonferroni (3 testes), o nível é 0,0167.\n")


def comparar_proporcoes(mask_a, mask_b, nome_a, nome_b):
    ya, yb = dados.loc[mask_a, "target"], dados.loc[mask_b, "target"]
    na, nb, ka, kb = len(ya), len(yb), ya.sum(), yb.sum()
    pa, pb = ka / na, kb / nb
    pool = (ka + kb) / (na + nb)
    z = (pa - pb) / np.sqrt(pool * (1 - pool) * (1 / na + 1 / nb))
    p = 2 * norm.sf(abs(z))
    ep = np.sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb)
    d = pa - pb
    print(f"{nome_a} ({pa * 100:.2f}%, n={na:,}) x {nome_b} ({pb * 100:.2f}%, n={nb:,}):\n"
          f"   diferença {d * 100:+.2f} p.p. (IC95% {(d - 1.96 * ep) * 100:+.2f} a "
          f"{(d + 1.96 * ep) * 100:+.2f}), z = {z:.2f}, p = {p:.4f}")


comparar_proporcoes(dados["produto_duravel"] == 0, dados["produto_duravel"] == 1,
                    "Não durável", "Durável")
comparar_proporcoes(dados["flag_atraso"] == 0, dados["flag_atraso"] == 1,
                    "Entrega no prazo", "Entrega atrasada")
comparar_proporcoes(dados["nota_primeira"] >= 4, dados["nota_primeira"] <= 2,
                    "Nota 4 ou 5", "Nota 1 ou 2")

titulo("CONCLUÍDO")
print("Arquivos gerados: tabela_logit_corrigida_v3.csv, linha_regressao_logistica_v3.csv")
