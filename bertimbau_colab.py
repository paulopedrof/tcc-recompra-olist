"""
================================================================================
ANÁLISE DE SENTIMENTO COM BERTimbau — Google Colab (GPU T4)
================================================================================
TCC — MBA em Data Science e Analytics — USP/Esalq
Autor: Paulo Pedro Filho

O QUE ESTE SCRIPT FAZ
O BERTimbau (Souza et al., 2020) é um modelo de linguagem pré-treinado em
português brasileiro, mas não classifica sentimento "de fábrica". Por isso ele
é ajustado (fine-tuning) com as próprias avaliações da Olist, usando a nota
como rótulo de polaridade:
    nota 1-2 → negativo | nota 3 → neutro | nota 4-5 → positivo

Para evitar que o modelo "decore" as avaliações em que treinou, usamos
validação cruzada em duas partes (cross-fitting):
    - treina na metade A e classifica a metade B
    - treina na metade B e classifica a metade A
Assim, toda avaliação é classificada por um modelo que nunca a viu.

SAÍDA
bertimbau_sentimento_reviews.csv — uma linha por avaliação, com as
probabilidades de sentimento negativo, neutro e positivo. Avaliações sem
texto ficam com probabilidades vazias (NaN) e tem_texto = 0.

COMO RODAR
1. Colab → Ambiente de execução → Alterar tipo → GPU T4
2. Cole este script inteiro numa célula e execute
3. Tempo estimado: 30 a 40 minutos. Mantenha a aba aberta.
4. Ao final, baixe bertimbau_sentimento_reviews.csv do Drive para a pasta
   do projeto no Mac (mesma pasta dos CSVs da Olist)
================================================================================
"""

# ── 1. Bibliotecas ──
import random
import time

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

# ── 2. Configurações ──
MODELO     = "neuralmind/bert-base-portuguese-cased"  # BERTimbau base oficial
MAX_LEN    = 128     # avaliações são curtas; 128 tokens cobrem quase todas
BATCH      = 32      # tamanho do lote no treino
EPOCAS     = 2       # passagens completas pelos dados de treino
LR         = 2e-5    # taxa de aprendizado padrão para fine-tuning de BERT
SEMENTE    = 42      # garante reprodutibilidade
N_PARTES   = 2       # validação cruzada em duas partes

ROTULOS    = {0: "negativo", 1: "neutro", 2: "positivo"}


def fixar_semente(s):
    """Fixa todas as fontes de aleatoriedade para resultados reproduzíveis."""
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


fixar_semente(SEMENTE)

# ── 3. Verifica a GPU ──
assert torch.cuda.is_available(), "GPU não encontrada — ative a GPU T4 no Colab"
device = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── 4. Monta o Google Drive e carrega as avaliações ──
from google.colab import drive
drive.mount("/content/drive")

path = "/content/drive/MyDrive/Colab Notebooks/TCC e-Commerce Brasil/"
reviews = pd.read_csv(path + "olist_order_reviews_dataset.csv")
print(f"Avaliações carregadas: {len(reviews):,}")

# ── 5. Texto consolidado (título + mensagem) ──
reviews["texto"] = (
    reviews["review_comment_title"].fillna("") + " " +
    reviews["review_comment_message"].fillna("")
).str.strip().replace("", np.nan)

reviews["tem_texto"] = reviews["texto"].notna().astype(int)

# Rótulo de polaridade a partir da nota (usado apenas para o fine-tuning)
mapa_nota = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}
reviews["rotulo"] = reviews["review_score"].map(mapa_nota)

com_texto = reviews[reviews["tem_texto"] == 1].reset_index(drop=True)
sem_texto = reviews[reviews["tem_texto"] == 0].reset_index(drop=True)

textos  = com_texto["texto"].tolist()
rotulos = com_texto["rotulo"].astype(int).tolist()

print(f"Com texto: {len(com_texto):,} | Sem texto: {len(sem_texto):,}")
print("Distribuição dos rótulos (avaliações com texto):")
print(com_texto["rotulo"].map(ROTULOS).value_counts())

# ── 6. Tokenizador do BERTimbau ──
tokenizer = AutoTokenizer.from_pretrained(MODELO)


def gerar_lotes(lista_textos, lista_rotulos=None, tamanho=BATCH, embaralhar=False):
    """Divide os textos em lotes e converte para o formato do BERTimbau."""
    indices = np.arange(len(lista_textos))
    if embaralhar:
        np.random.shuffle(indices)
    for inicio in range(0, len(indices), tamanho):
        lote = indices[inicio:inicio + tamanho]
        enc = tokenizer(
            [lista_textos[i] for i in lote],
            padding=True, truncation=True,
            max_length=MAX_LEN, return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        y = None
        if lista_rotulos is not None:
            y = torch.tensor([lista_rotulos[i] for i in lote], device=device)
        yield enc, y


def criar_scaler():
    """Escalonador para precisão mista (fp16) — compatível com versões do PyTorch."""
    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler()


# ── 7. Fine-tuning ──
def treinar(lista_textos, lista_rotulos):
    """
    Ajusta o BERTimbau para classificar sentimento em 3 classes.
    O aviso de 'pesos recém-inicializados' no carregamento é esperado:
    é a camada de classificação que será treinada aqui.
    """
    fixar_semente(SEMENTE)
    modelo = AutoModelForSequenceClassification.from_pretrained(
        MODELO, num_labels=3
    ).to(device)

    otimizador = torch.optim.AdamW(modelo.parameters(), lr=LR, weight_decay=0.01)
    lotes_por_epoca = int(np.ceil(len(lista_textos) / BATCH))
    total_passos = EPOCAS * lotes_por_epoca
    agenda = get_linear_schedule_with_warmup(
        otimizador, int(0.1 * total_passos), total_passos
    )
    scaler = criar_scaler()

    modelo.train()
    for epoca in range(EPOCAS):
        perda_total, n = 0.0, 0
        for enc, y in tqdm(gerar_lotes(lista_textos, lista_rotulos, embaralhar=True),
                           total=lotes_por_epoca, desc=f"Época {epoca + 1}/{EPOCAS}"):
            otimizador.zero_grad()
            with torch.autocast("cuda", dtype=torch.float16):
                saida = modelo(**enc, labels=y)
            scaler.scale(saida.loss).backward()
            scaler.unscale_(otimizador)
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            scaler.step(otimizador)
            scaler.update()
            agenda.step()
            perda_total += saida.loss.item()
            n += 1
        print(f"  Época {epoca + 1}: perda média = {perda_total / n:.4f}")
    return modelo


# ── 8. Classificação (inferência) ──
@torch.no_grad()
def classificar(modelo, lista_textos):
    """Retorna as probabilidades [negativo, neutro, positivo] de cada texto."""
    modelo.eval()
    probs = []
    total = int(np.ceil(len(lista_textos) / (BATCH * 2)))
    for enc, _ in tqdm(gerar_lotes(lista_textos, tamanho=BATCH * 2),
                       total=total, desc="Classificando"):
        with torch.autocast("cuda", dtype=torch.float16):
            logits = modelo(**enc).logits
        probs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.vstack(probs)


# ── 9. Validação cruzada em duas partes (cross-fitting) ──
inicio = time.time()
probs_oof = np.zeros((len(textos), 3))
parte_de  = np.zeros(len(textos), dtype=int)

skf = StratifiedKFold(n_splits=N_PARTES, shuffle=True, random_state=SEMENTE)

for k, (idx_treino, idx_prever) in enumerate(skf.split(textos, rotulos)):
    print(f"\n=== Parte {k + 1}/{N_PARTES}: treina em {len(idx_treino):,} "
          f"e classifica {len(idx_prever):,} avaliações ===")
    modelo = treinar([textos[i] for i in idx_treino],
                     [rotulos[i] for i in idx_treino])
    probs_oof[idx_prever] = classificar(modelo, [textos[i] for i in idx_prever])
    parte_de[idx_prever] = k
    del modelo
    torch.cuda.empty_cache()

print(f"\nTempo total: {(time.time() - inicio) / 60:.1f} minutos")

# ── 10. Qualidade do classificador (fora da amostra) ──
# Métrica para o TCC: quão bem o BERTimbau ajustado reconhece a polaridade
# em avaliações que nunca viu
pred = probs_oof.argmax(axis=1)
print("\nDesempenho do BERTimbau ajustado (avaliações fora da amostra):")
print(classification_report(rotulos, pred,
                            target_names=list(ROTULOS.values()), digits=3))
print(f"F1 macro: {f1_score(rotulos, pred, average='macro'):.3f}")

# ── 11. Monta e salva o arquivo de saída ──
com_texto["prob_neg"]   = probs_oof[:, 0]
com_texto["prob_neu"]   = probs_oof[:, 1]
com_texto["prob_pos"]   = probs_oof[:, 2]
com_texto["label_bert"] = pd.Series(pred).map(ROTULOS).values
com_texto["parte"]      = parte_de

for col in ["prob_neg", "prob_neu", "prob_pos", "label_bert", "parte"]:
    sem_texto[col] = np.nan

colunas = ["review_id", "order_id", "review_score", "review_creation_date",
           "tem_texto", "prob_neg", "prob_neu", "prob_pos", "label_bert", "parte"]

saida = pd.concat([com_texto[colunas], sem_texto[colunas]], ignore_index=True)
arquivo = path + "bertimbau_sentimento_reviews.csv"
saida.to_csv(arquivo, index=False)

print(f"\nArquivo salvo: {arquivo}")
print(f"Linhas: {len(saida):,} | Com texto: {int(saida['tem_texto'].sum()):,}")
print("\nPróximo passo: baixar bertimbau_sentimento_reviews.csv para a pasta "
      "do projeto no Mac e rodar o pipeline v3.")
