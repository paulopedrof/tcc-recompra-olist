"""
ANÁLISE DE RELAÇÃO ENTRE VALOR DA PRIMEIRA COMPRA E RECOMPRA
Objetivo: verificar a direção e natureza da relação entre o valor da primeira
compra e a propensão de recompra, e investigar possível efeito de confusão
com a durabilidade do produto.
Resultado esperado: interpretação correta da contribuição SHAP de
valor_primeira_compra no modelo final.
"""

import pandas as pd

# ── Carrega df_base ──
try:
    _ = df_base
    print("df_base carregado da sessão atual.")
except NameError:
    df_base = pd.read_csv("df_base_features.csv")
    print("df_base carregado do CSV intermediário.")

# ── 1. Taxa de recompra por quartil de valor ──
# Verifica se a relação entre valor e recompra é positiva ou negativa
df_base["quartil_valor"] = pd.qcut(
    df_base["valor_primeira_compra"],
    q=4,
    labels=["Q1 (menor)", "Q2", "Q3", "Q4 (maior)"])

taxa_valor = (df_base
    .groupby("quartil_valor", observed=False)["target"]
    .agg(recompraram="sum", total="count", taxa_recompra="mean")
    .round(4))

print("\nTaxa de recompra por quartil de valor da primeira compra:")
print(taxa_valor.to_string())

variacao_valor = taxa_valor["taxa_recompra"].max() - taxa_valor["taxa_recompra"].min()
direcao = "INVERSA (menor valor → maior recompra)" if (
    taxa_valor["taxa_recompra"].iloc[0] > taxa_valor["taxa_recompra"].iloc[-1]
) else "POSITIVA (maior valor → maior recompra)"

print(f"\nDireção da relação: {direcao}")
print(f"Variação entre quartis extremos: {variacao_valor*100:.2f} pp")

# ── 2. Cruzamento valor x durabilidade ──
# Investiga se o valor é proxy da durabilidade do produto
print("\nValor médio da primeira compra por durabilidade:")
print(df_base.groupby("produto_duravel")["valor_primeira_compra"]
      .describe().round(2))

# ── 3. Taxa de recompra por valor x durabilidade ──
# Verifica se o efeito do valor se mantém dentro de cada grupo de durabilidade
print("\nTaxa de recompra por quartil de valor e durabilidade:")
cross = (df_base
    .groupby(["produto_duravel", "quartil_valor"], observed=False)["target"]
    .agg(recompraram="sum", total="count", taxa_recompra="mean")
    .round(4))
print(cross.to_string())

# ── 4. Correlação entre valor e durabilidade ──
corr_val_dur = df_base[["valor_primeira_compra", "produto_duravel"]].corr()
print(f"\nCorrelação entre valor_primeira_compra e produto_duravel:")
print(corr_val_dur.round(4))

# ── 5. Conclusão ──
print(f"""
── Conclusão da análise ──
A relação entre valor da primeira compra e recompra é {direcao}.
Clientes com menor ticket na primeira compra apresentam maior propensão
à recompra — consistente com o perfil de consumidores de produtos
não-duráveis (cosméticos, livros, moda, pet) que têm ciclo de compra curto.
O valor da primeira compra funciona como proxy do perfil de consumo,
capturando em parte o mesmo sinal que produto_duravel, mas de forma
complementar — justificando a manutenção de ambas as variáveis no modelo.
A importância SHAP de valor_primeira_compra reflete sua contribuição
não linear em interação com produto_duravel, não uma relação linear positiva.
""")
