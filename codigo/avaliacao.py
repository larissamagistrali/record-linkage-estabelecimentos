"""Avalia o match com a amostra rotulada, ponderando cada estrato de score pelo seu tamanho na população.

Rótulos I (incerto) ficam fora das métricas.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
DIR_SAIDA = RAIZ / "resultados" / "avaliacao"
FAIXAS_SCORE = [0, .5, .6, .7, .75, .8, .85, .9, .95, 1.01]  # as mesmas da amostra
Z = 1.96


def wilson(sucessos, n):
    """Intervalo de confiança de 95% de Wilson para uma proporção."""
    if n == 0:
        return np.nan, np.nan
    p = sucessos / n
    centro = (p + Z**2 / (2 * n)) / (1 + Z**2 / n)
    margem = Z * np.sqrt(p * (1 - p) / n + Z**2 / (4 * n**2)) / (1 + Z**2 / n)
    return centro - margem, centro + margem


def carregar():
    rot = pd.read_csv(RAIZ / "rotulagem" / "amostra_cega.csv", sep=";", encoding="utf-8-sig", dtype=str)
    rot["rotulo"] = rot.rotulo.str.strip().str.upper()
    chave = pd.read_csv(RAIZ / "rotulagem" / "chave_amostra.csv", dtype={"token_fechado": str})
    rot = rot[["id_par", "rotulo", "observacao"]].astype({"id_par": int}).merge(chave, on="id_par")

    r = pd.read_parquet(RAIZ / "resultados" / "match_completo.parquet")
    r = r[r.score > 0].copy()
    r["estrato"] = pd.cut(r.score, FAIXAS_SCORE, right=False)

    am = rot.merge(r, on="id_aberto", suffixes=("", "_atual"))
    divergentes = (am.token_fechado != am.token_fechado_atual).sum()
    assert divergentes == 0, f"{divergentes} pares mudaram desde a amostragem: rode a avaliação sem rerodar o pipeline"

    validos = am[am.rotulo.isin(["S", "N"])].copy()
    validos["y"] = (validos.rotulo == "S").astype(int)
    pop = r.estrato.value_counts()
    amostra = validos.estrato.value_counts()
    validos["peso"] = validos.estrato.map(pop / amostra).astype(float)
    return am, validos, r


def por_estrato(am, validos, r):
    t = (
        am.groupby("estrato", observed=True)
        .rotulo.value_counts().unstack(fill_value=0)
        .reindex(columns=["S", "N", "I"], fill_value=0)
    )
    t["pares_populacao"] = r.estrato.value_counts()
    t["prop_S"] = t.S / (t.S + t.N)
    ic = [wilson(s, s + n) for s, n in zip(t.S, t.N)]
    t["ic95_inf"] = [i[0] for i in ic]
    t["ic95_sup"] = [i[1] for i in ic]
    t["S_estimados_pop"] = (t.prop_S * t.pares_populacao).round()
    return t


def por_faixa_decisao(validos, r):
    """Calcula as métricas de cada faixa de decisão do pipeline."""
    linhas = []
    for faixa, g in validos.groupby("faixa"):
        s, n = g.y.sum(), len(g)
        p_s = np.average(g.y, weights=g.peso)
        inf, sup = wilson(s, n)
        linhas.append({"faixa": faixa, "rotulados": n, "S": s, "N": n - s,
                       "prop_S_ponderada": p_s, "ic95_inf_nao_ponderado": inf, "ic95_sup_nao_ponderado": sup,
                       "estab_populacao": (r.faixa == faixa).sum()})
    t = pd.DataFrame(linhas).set_index("faixa")
    t["S_estimados"] = (t.prop_S_ponderada * t.estab_populacao).round()
    t["N_estimados"] = t.estab_populacao - t.S_estimados
    return t


def limiares(validos, r):
    """Calcula precisão, recall e erros estimados para cada limiar de credenciado."""
    total_s = np.sum(validos.y * validos.peso)
    linhas = []
    for lim in np.arange(0.75, 1.0001, 0.01):
        acima = validos[validos.score >= lim]
        if acima.empty:
            continue
        precisao = np.average(acima.y, weights=acima.peso)
        recall = np.sum(acima.y * acima.peso) / total_s
        n_pop = (r.score >= lim).sum()
        linhas.append({
            "limiar": round(lim, 2),
            "precisao": precisao,
            "recall": recall,
            "f1": 2 * precisao * recall / (precisao + recall),
            "declarados_credenciados": n_pop,
            # N acima do limiar: não credenciados escondidos do ranking
            "oportunidades_perdidas_est": round((1 - precisao) * n_pop),
            # S abaixo do limiar: credenciados enviados ao ranking
            "contatos_desnecessarios_est": round(np.sum(validos.y[validos.score < lim] * validos.peso[validos.score < lim])),
        })
    return pd.DataFrame(linhas)


def main():
    DIR_SAIDA.mkdir(parents=True, exist_ok=True)
    am, validos, r = carregar()
    print(f"rotulados: {len(am)}  S: {(am.rotulo == 'S').sum()}  N: {(am.rotulo == 'N').sum()}  "
          f"I: {(am.rotulo == 'I').sum()}  (I fora das métricas)\n")

    saidas = {
        "por_estrato": por_estrato(am, validos, r),
        "por_faixa_decisao": por_faixa_decisao(validos, r),
        "limiares": limiares(validos, r),
    }
    pd.set_option("display.width", 220)
    for nome, t in saidas.items():
        t.to_csv(DIR_SAIDA / f"{nome}.csv")
        print(f"── {nome}\n{t.round(3).to_string()}\n")

    erros = am[am.rotulo.isin(["S", "N"])]
    erros = erros[((erros.faixa == "credenciado") & (erros.rotulo == "N"))
                  | ((erros.faixa == "nao_credenciado") & (erros.rotulo == "S"))]
    erros[["id_par", "rotulo", "faixa", "score", "exemplo_nome", "cidade_bloco", "nome_fechado", "observacao"]] \
        .sort_values("score").to_csv(DIR_SAIDA / "erros.csv", index=False)
    print(f"── erros nas faixas extremas (resultados/avaliacao/erros.csv)\n"
          f"{erros[['rotulo', 'faixa', 'score', 'exemplo_nome', 'nome_fechado']].sort_values('score').to_string(index=False)}")


if __name__ == "__main__":
    main()
