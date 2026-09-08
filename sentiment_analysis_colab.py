"""
ANÁLISE DE SENTIMENTO VIA BERTimbau — Google Colab
===================================================
Este script deve ser executado no Google Colab com GPU Tesla T4 ativa.
Configuração: Runtime → Alterar tipo de ambiente de execução → GPU T4

Objetivo: aplicar análise de sentimento multilíngue às avaliações textuais
dos clientes da base Olist, gerando features de sentimento para uso no
modelo de predição de recompra.

Modelo utilizado: distilbert-base-multilingual-cased-sentiments-student
Fonte: Hugging Face — https://huggingface.co/lxyuan/distilbert-base-multilingual-cased-sentiments-student
Justificativa: modelo multilíngue com suporte nativo ao português brasileiro,
classificação em três classes (positivo, negativo, neutro) e desempenho
competitivo com modelos maiores em textos curtos de avaliação.

Output: sentiment_features.csv — features de sentimento agregadas por cliente,
para download e incorporação no pipeline principal (tcc_completo.py).
"""

# ── 1. Verifica disponibilidade da GPU ──
# Confirma que o ambiente Colab está configurado com aceleração por GPU
# Necessário para processamento eficiente das 42.687 avaliações com texto
import torch

print(f"CUDA disponível: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# ── 2. Instala bibliotecas necessárias ──
# transformers: biblioteca Hugging Face para modelos de linguagem pré-treinados
# torch: backend de deep learning
# !pip install transformers torch -q

# ── 3. Monta o Google Drive ──
# Os arquivos CSV da Olist devem estar previamente carregados no Drive
from google.colab import drive
drive.mount('/content/drive')

# ── 4. Define o caminho dos arquivos ──
# Ajustar conforme a estrutura de pastas do usuário no Google Drive
path = "/content/drive/MyDrive/Colab Notebooks/TCC e-Commerce Brasil/"

# ── 5. Carrega as tabelas necessárias ──
# Apenas as tabelas relevantes para o processamento de sentimento
import pandas as pd

reviews   = pd.read_csv(path + "olist_order_reviews_dataset.csv")
orders    = pd.read_csv(path + "olist_orders_dataset.csv")
customers = pd.read_csv(path + "olist_customers_dataset.csv")

print(f"reviews:   {reviews.shape}")
print(f"orders:    {orders.shape}")
print(f"customers: {customers.shape}")

# ── 6. Consolidação do texto das avaliações ──
# Combina título e mensagem da avaliação num único campo de texto
# Quando o cliente preencheu apenas um dos campos, usa o disponível
# Avaliações sem texto receberão sentimento derivado do score numérico (etapa 9)
reviews["texto"] = (
    reviews["review_comment_title"].fillna("") + " " +
    reviews["review_comment_message"].fillna("")
).str.strip().replace("", None)

print(f"\nReviews com texto consolidado: {reviews['texto'].notna().sum():,}")
print(f"Reviews sem texto:             {reviews['texto'].isna().sum():,}")

# ── 7. Carrega o pipeline de análise de sentimento ──
# Modelo pré-treinado multilíngue com classificação em três classes:
# positive, negative, neutral
# device=0 direciona o processamento para a GPU Tesla T4
from transformers import pipeline

sentiment_pipe = pipeline(
    "sentiment-analysis",
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    device=0
)

# Teste de sanidade com exemplos em português
teste = [
    "Produto excelente, chegou antes do prazo!",
    "Pessimo produto, veio quebrado e o vendedor nao respondeu.",
    "Ok, nada demais."
]

print("\nTeste do modelo:")
for t in teste:
    resultado = sentiment_pipe(t)
    print(f"  Texto: {t[:50]}")
    print(f"  Resultado: {resultado}\n")

# ── 8. Processamento em lote das avaliações com texto ──
# Batch size de 64 otimiza o uso da memória da GPU
# truncation=True garante que textos longos não ultrapassem o limite do modelo (512 tokens)
from tqdm import tqdm

reviews_com_texto = reviews[reviews["texto"].notna()].copy()
reviews_sem_texto = reviews[reviews["texto"].isna()].copy()

textos     = reviews_com_texto["texto"].tolist()
batch_size = 64
resultados = []

for i in tqdm(range(0, len(textos), batch_size)):
    batch = textos[i:i + batch_size]
    saida = sentiment_pipe(batch, truncation=True, max_length=512)
    resultados.extend(saida)

# Extrai label (classe) e score (confiança) de cada predição
reviews_com_texto["sentimento_label"] = [r["label"] for r in resultados]
reviews_com_texto["sentimento_score"] = [r["score"] for r in resultados]

print(f"\nProcessadas: {len(reviews_com_texto):,} reviews com texto")
print(f"\nDistribuição de sentimento:")
print(reviews_com_texto["sentimento_label"].value_counts())

# ── 9. Imputação de sentimento para avaliações sem texto ──
# Para as 56.537 avaliações sem texto descritivo, o sentimento é derivado
# do score numérico (1 a 5), que constitui proxy válido de satisfação
# Score 1-2 → negativo | Score 3 → neutro | Score 4-5 → positivo
# Score de confiança fixado em 0,7 — valor conservador abaixo da média
# dos classificados pelo modelo (0,68), sinalizando menor certeza
def score_para_sentimento(score):
    if pd.isna(score):  return "neutral"
    elif score <= 2:    return "negative"
    elif score == 3:    return "neutral"
    else:               return "positive"

reviews_sem_texto["sentimento_label"] = (reviews_sem_texto["review_score"]
                                          .apply(score_para_sentimento))
reviews_sem_texto["sentimento_score"] = 0.7

# ── 10. Consolidação das duas partes ──
reviews_completo = pd.concat(
    [reviews_com_texto, reviews_sem_texto], ignore_index=True)

print(f"\nTotal de reviews processadas: {len(reviews_completo):,}")
print(f"\nDistribuição final de sentimento:")
print(reviews_completo["sentimento_label"].value_counts())
print(f"\nProporção:")
print(reviews_completo["sentimento_label"].value_counts(normalize=True).round(3))

# ── 11. Associa cada review ao customer_unique_id ──
# Realiza joins entre reviews → orders → customers para obter
# o identificador persistente do cliente (customer_unique_id)
reviews_com_cliente = (reviews_completo
    .merge(orders[["order_id", "customer_id"]], on="order_id", how="left")
    .merge(customers[["customer_id", "customer_unique_id"]],
           on="customer_id", how="left"))

print(f"\nNulos em customer_unique_id: "
      f"{reviews_com_cliente['customer_unique_id'].isnull().sum():,}")

# ── 12. Agrega features de sentimento por cliente ──
# Para cada cliente, calcula métricas agregadas sobre todas as suas avaliações:
# - review_score_medio: score numérico médio (1 a 5)
# - sentimento_score_medio: confiança média do modelo
# - prop_positivo/negativo/neutro: proporção de cada classe de sentimento
# - n_reviews: número total de avaliações registradas
# Nota: essas features são calculadas sobre TODAS as avaliações do cliente.
# No pipeline principal (tcc_completo.py), apenas a avaliação da primeira
# compra é utilizada como feature preditora, para evitar data leakage.
# Este arquivo serve como referência do processamento completo de sentimento.
sentimento_cliente = (reviews_com_cliente
    .groupby("customer_unique_id")
    .agg(
        review_score_medio     = ("review_score",      "mean"),
        sentimento_score_medio = ("sentimento_score",  "mean"),
        prop_positivo          = ("sentimento_label",
                                  lambda x: (x == "positive").mean()),
        prop_negativo          = ("sentimento_label",
                                  lambda x: (x == "negative").mean()),
        prop_neutro            = ("sentimento_label",
                                  lambda x: (x == "neutral").mean()),
        n_reviews              = ("review_id",         "count")
    )
    .reset_index())

print(f"\nShape do output: {sentimento_cliente.shape}")
print(f"\nEstatísticas:")
print(sentimento_cliente.describe().round(3))

# ── 13. Salva o output no Google Drive ──
# O arquivo sentiment_features.csv deve ser baixado e colocado na mesma
# pasta dos CSVs da Olist para uso no pipeline principal (tcc_completo.py)
output_path = path + "sentiment_features.csv"
sentimento_cliente.to_csv(output_path, index=False)

print(f"\nArquivo salvo em: {output_path}")
print(f"Shape final: {sentimento_cliente.shape}")
print(f"Tamanho: {sentimento_cliente.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
print("\nPróximo passo: baixar sentiment_features.csv e copiar para a pasta")
print("dos CSVs da Olist antes de executar tcc_completo.py")
