"""Compara métodos de similaridade com a amostra rotulada, usando métricas ponderadas por estrato.

A tabela principal traz quatro métodos.
"""
import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from avaliacao import DIR_SAIDA, carregar

MODELOS_EMBEDDING = {
    "emb_minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "emb_mpnet": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
}
LEXICAS = ["tfidf", "jaro_winkler", "token_sort", "token_set", "score_nucleo"]
METODOS_PRINCIPAIS = {
    "tfidf": "TF-IDF de n-gramas de caracteres",
    "score": "Escore combinado (proposto)",
    "logistica_lexica": "Regressão logística",
    "emb_mpnet": "Embedding MPNet multilíngue",
}
NOMES_APENDICE = {
    "jaro_winkler": "Jaro-Winkler",
    "token_sort": "Token sort ratio",
    "token_set": "Token set ratio",
    "score_nucleo": "Núcleo do nome",
    "emb_minilm": "Embedding MiniLM multilíngue",
    "logistica_lexica_emb": "Regressão logística com embeddings",
}
REFERENCIA = "tfidf"
N_BOOTSTRAP = 2000
SEMENTE = 42


def similaridade_embeddings(validos):
    from sentence_transformers import SentenceTransformer

    for coluna, nome in MODELOS_EMBEDDING.items():
        modelo = SentenceTransformer(nome)
        a = modelo.encode(validos.nome_norm.tolist(), normalize_embeddings=True, batch_size=64)
        f = modelo.encode(validos.nome_norm_fechado.tolist(), normalize_embeddings=True, batch_size=64)
        validos[coluna] = (a * f).sum(axis=1)
    return validos


def preparar_features(validos, colunas):
    x = validos[colunas].copy()
    if "score_nucleo" in colunas:
        # núcleo ausente vira um indicador
        x["nucleo_ausente"] = x.score_nucleo.isna().astype(int)
        x["score_nucleo"] = x.score_nucleo.fillna(x.score_nucleo.median())
    return x.to_numpy()


def logistica_validacao_cruzada(validos, colunas, repeticoes=20):
    """Retorna as probabilidades fora da amostra por validação cruzada repetida e os coeficientes."""
    x = preparar_features(validos, colunas)
    y = validos.y.to_numpy()
    peso = validos.peso.to_numpy()
    soma = np.zeros(len(y))
    contagem = np.zeros(len(y))
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeticoes, random_state=SEMENTE)
    for treino, teste in cv.split(x, y):
        modelo = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        modelo.fit(x[treino], y[treino], logisticregression__sample_weight=peso[treino])
        soma[teste] += modelo.predict_proba(x[teste])[:, 1]
        contagem[teste] += 1
    # coeficientes do modelo ajustado em todos os dados
    final = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    final.fit(x, y, logisticregression__sample_weight=peso)
    nomes = colunas + (["nucleo_ausente"] if "score_nucleo" in colunas else [])
    coefs = pd.Series(final[-1].coef_[0], index=nomes)
    return soma / contagem, coefs


def metricas(y, s, peso):
    return roc_auc_score(y, s, sample_weight=peso), average_precision_score(y, s, sample_weight=peso)


def bootstrap_diferenca(y, s, s_ref, peso, rng):
    """Calcula o intervalo de 95% da diferença de precisão média para a referência por bootstrap."""
    n = len(y)
    difs = []
    for _ in range(N_BOOTSTRAP):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        difs.append(average_precision_score(y[i], s[i], sample_weight=peso[i])
                    - average_precision_score(y[i], s_ref[i], sample_weight=peso[i]))
    return np.percentile(difs, [2.5, 97.5])


def precisao_no_recall(y, s, peso, recall_alvo=0.70):
    """Calcula a precisão no ponto em que o recall atinge o valor alvo."""
    ordem = np.argsort(-s)
    y, peso = y[ordem], peso[ordem]
    tp = np.cumsum(y * peso)
    recall = tp / tp[-1]
    k = np.argmax(recall >= recall_alvo)
    return tp[k] / np.cumsum(peso)[k]


def main():
    _, validos, _ = carregar()
    validos = similaridade_embeddings(validos.reset_index(drop=True))

    candidatos = {c: validos[c].fillna(0).to_numpy() for c in
                  LEXICAS + ["score"] + list(MODELOS_EMBEDDING)}
    coeficientes = {}
    configuracoes = {
        "logistica_lexica": LEXICAS,
        "logistica_lexica_emb": LEXICAS + list(MODELOS_EMBEDDING),
    }
    for nome, colunas in configuracoes.items():
        candidatos[nome], coeficientes[nome] = logistica_validacao_cruzada(validos, colunas)

    y = validos.y.to_numpy()
    peso = validos.peso.to_numpy()
    rng = np.random.default_rng(SEMENTE)
    linhas = []
    for nome, s in candidatos.items():
        auc, ap = metricas(y, s, peso)
        inf, sup = (np.nan, np.nan) if nome == REFERENCIA else bootstrap_diferenca(y, s, candidatos[REFERENCIA], peso, rng)
        linhas.append({"medida": nome, "auc_roc": auc, "precisao_media": ap,
                       "precisao_recall70": precisao_no_recall(y, s, peso),
                       f"dif_pm_vs_{REFERENCIA}_ic95_inf": inf, f"dif_pm_vs_{REFERENCIA}_ic95_sup": sup})
    tabela = pd.DataFrame(linhas).sort_values("precisao_media", ascending=False)
    tabela.insert(1, "metodo", tabela.medida.map({**METODOS_PRINCIPAIS, **NOMES_APENDICE}))
    principal = tabela[tabela.medida.isin(METODOS_PRINCIPAIS)]
    principal.to_csv(DIR_SAIDA / "comparacao_principal.csv", index=False)
    tabela.to_csv(DIR_SAIDA / "comparacao_apendice.csv", index=False)
    coeficientes["logistica_lexica"].to_csv(DIR_SAIDA / "coeficientes_logistica.csv", header=["coeficiente"])
    # probabilidades fora da amostra da regressão, usadas nas curvas de precisão e recall
    for nome in configuracoes:
        validos[nome] = candidatos[nome]
    validos.to_csv(DIR_SAIDA / "pares_rotulados_com_scores.csv", index=False)

    pd.set_option("display.width", 220)
    print("── tabela principal")
    print(principal.drop(columns="medida").round(3).to_string(index=False))
    print("\n── apêndice (todos os métodos avaliados)")
    print(tabela.drop(columns="medida").round(3).to_string(index=False))
    print("\ncoeficientes padronizados da regressão logística (ajustada em toda a amostra):")
    print(coeficientes["logistica_lexica"].round(2).to_string())

    # pares em que embedding e TF-IDF mais discordam
    validos["dif_emb"] = validos.emb_mpnet - validos.tfidf
    cols = ["rotulo", "nome_norm", "nome_norm_fechado", "tfidf", "emb_mpnet"]
    print("\nembedding muito acima do TF-IDF:")
    print(validos.nlargest(8, "dif_emb")[cols].round(2).to_string(index=False))
    print("\nembedding muito abaixo do TF-IDF:")
    print(validos.nsmallest(8, "dif_emb")[cols].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
