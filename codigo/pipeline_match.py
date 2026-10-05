"""Identifica estabelecimentos do arranjo aberto ainda não credenciados e os prioriza por valor e transações.

Falso negativo é preferível a falso positivo, por isso o limiar para declarar credenciado é alto.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler
from sklearn.feature_extraction.text import TfidfVectorizer
from unidecode import unidecode

RAIZ = Path(__file__).resolve().parent.parent
DIR_DADOS = RAIZ / "dataset"
DIR_SAIDA = RAIZ / "resultados"

TAM_NOME_ABERTO = 22  # nome truncado no arranjo aberto
TAM_CIDADE_ABERTO = 13  # cidade truncada no arranjo aberto
TOP_K = 5  # candidatos por estabelecimento do arranjo aberto
LIMIAR_CREDENCIADO = 0.93  # a partir deste score o estabelecimento é considerado credenciado
LIMIAR_REVISAO = 0.80  # entre este valor e o anterior vai para revisão
LIMIAR_CIDADE = 88  # similaridade mínima para reconciliar grafias de cidade
MIN_PORTADORES = 3  # mínimo de portadores para entrar no ranking principal
RAZOES_POR_UNIDADE_REDE = 0.5  # proporção máxima de razões sociais por unidade para ser rede

# sufixos societários, só no fim e isolados
_SUFIXOS = r"(LTDA|LTD|ME|MEI|EPP|EIRELI|SA|S A|S/A|CIA|COMPANHIA|SOCIEDADE|EMPRESARIA|INDIVIDUAL)"
_RE_SUFIXO = re.compile(rf"\s+{_SUFIXOS}\s*$")
# numeração de loja no fim
_RE_LOJA = re.compile(r"\s*[-–]?\s*(LOJA|FILIAL|UNIDADE|UND|LJ|FL)?\s*\d{1,5}\s*$")
# prefixo de subadquirente ou facilitador, como MP* e PAG*
_RE_PREFIXO = re.compile(r"^[A-Z0-9 .]{1,14}\*\s*")
_RE_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])")
# raiz de CNPJ no início do nome, padrão de MEI
_RE_CNPJ_RAIZ = re.compile(r"^\s*\d{2}\.?\d{3}\.?\d{3}\s*")
# pagamento por QR code do facilitador, como MP*QRNOMEDALOJA
_RE_QR = re.compile(r"\*\s*QR(?=[A-Z]{3})")
# no Mercado Pago a cidade informada costuma ser a da sede, não a da loja
_RE_MERCADO_PAGO = re.compile(r"^(MP|MERCADO ?PAGO)\s*\*")
_CIDADE_SEDE_MERCADO_PAGO = "OSASCO"
_RE_NAO_ALFANUM = re.compile(r"[^A-Z0-9/ ]+")

# termos de categoria, que não identificam o estabelecimento
_GENERICOS = {
    "SUPERMERCADO", "SUPERMERCADOS", "SUPERMERC", "SUPERMERCAD", "SUPER", "MERCADO", "MERCADOS",
    "MINIMERCADO", "MINIMERCADOS", "MERCEARIA", "HIPERMERCADO", "ATACADISTA", "ATACADO",
    "ATACAREJO", "RESTAURANTE", "REST", "LANCHONETE", "LANCHES", "PADARIA", "CONFEITARIA",
    "PANIFICADORA", "ACOUGUE", "FARMACIA", "DROGARIA", "COMERCIO", "COM", "COMERCIAL",
    "ALIMENTOS", "ALIMENTACAO", "PRODUTOS", "PROD", "DE", "DA", "DO", "DOS", "DAS", "E",
}

# intermediadores: aplicativos de entrega e de mobilidade
_RESIDUOS = r"PENDING|RIDES|TRIP|EATS|CLUB|HELP|PAY|APP"
_RE_RESIDUOS = re.compile(rf"\b({_RESIDUOS})\b")
# o que sobra quando o nome era só o aplicativo
_RESTOS_APP = {"PENDING", "RIDES", "CLUB", "TRIP", "PAY", "APP", ""}

_MARCAS_ENTREGA = ("IFOOD", "IFD", "99FOOD", "99 FOOD", "RAPPI", "EATS")
_MARCAS_MOBILIDADE = ("UBER", "99APP", "99 APP", "99POP", "99PAY", "99TAXI")


def _re_marcas(marcas):
    # a marca não pode vir colada a outra letra e EATS precisa estar isolada
    partes = [rf"(?<![A-Z0-9]){re.escape(m)}" + (r"(?![A-Z])" if m == "EATS" else "") for m in marcas]
    return re.compile("|".join(partes))


_RE_ENTREGA = _re_marcas(_MARCAS_ENTREGA)
_RE_MOBILIDADE = _re_marcas(_MARCAS_MOBILIDADE)


def _base(texto):
    """Converte para maiúsculas sem acento e separa palavras em camelCase."""
    if not isinstance(texto, str):
        return ""
    return unidecode(_RE_CAMEL.sub(" ", texto)).upper().strip()


def normalizar_nome(texto):
    t = _RE_QR.sub("*", _base(texto))
    t = _RE_PREFIXO.sub("", t).replace("&", " E ")
    t = _RE_CNPJ_RAIZ.sub("", t)
    t = _RE_LOJA.sub("", t)
    t = _RE_NAO_ALFANUM.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    # remove sufixos encadeados, como LTDA ME
    anterior = None
    while anterior != t:
        anterior = t
        t = _RE_SUFIXO.sub("", t).strip()
        t = _RE_LOJA.sub("", t).strip()
    return t.replace("/", " ").strip()


def classificar_intermediador(texto):
    """Retorna a família do intermediador e o estabelecimento embutido no nome, se houver."""
    t = _base(texto)
    if _RE_MOBILIDADE.search(t):
        return "mobilidade", ""
    if _RE_ENTREGA.search(t):
        resto = t.rsplit("*", 1)[-1] if "*" in t else t
        # marca repetida após o asterisco indica a própria empresa do aplicativo
        if "*" in t and _RE_ENTREGA.search(resto):
            return "entrega", ""
        resto = _RE_ENTREGA.sub(" ", resto)
        resto = _RE_RESIDUOS.sub(" ", resto)
        resto = normalizar_nome(resto)
        if resto in _RESTOS_APP or len(resto) < 3:
            return "entrega", ""
        return "entrega", resto
    return None, None


def normalizar_cidade(texto):
    t = _RE_NAO_ALFANUM.sub(" ", _base(texto))
    return re.sub(r"\s+", " ", t).strip()[:TAM_CIDADE_ABERTO].strip()


def nucleo(nome):
    """Remove os termos de categoria do nome."""
    return " ".join(p for p in nome.split() if p not in _GENERICOS)


def reconciliar_cidades(ab, fe):
    """Mapeia grafias de cidade do arranjo aberto para as cidades do fechado."""
    conhecidas = list(fe.cidade_bloco.unique())
    mapa = {}
    for c in set(ab.cidade_bloco) - set(conhecidas):
        if len(c) < 5:
            continue
        m = process.extractOne(c, conhecidas, scorer=fuzz.ratio, score_cutoff=LIMIAR_CIDADE)
        if m:
            mapa[c] = m[0]
    print(f"cidades reconciliadas por similaridade: {len(mapa)}")
    ab["cidade_bloco"] = ab.cidade_bloco.replace(mapa)
    return ab


def preparar_aberto():
    a = pd.read_parquet(DIR_DADOS / "arranjo_aberto_transacoes_2026_01_2026_06.parquet")
    a = a[
        (a.tipo_evento == "TXA")
        & (a.tipo_transacao == "D")
        & (a.aprovada == 1)
        & (a.cancelada == 0)
        & (a.data_compra >= "2026-01-01")
        & (a.data_compra < "2026-07-01")
    ].copy()

    nomes = a[["nome_estabelecimento"]].drop_duplicates()
    inter = nomes.nome_estabelecimento.map(classificar_intermediador)
    nomes["intermediador"] = inter.str[0]
    nomes["nome_norm"] = [
        resto if fam == "entrega" else normalizar_nome(n)
        for n, (fam, resto) in zip(nomes.nome_estabelecimento, inter)
    ]
    nomes.loc[nomes.intermediador == "mobilidade", "nome_norm"] = ""
    a = a.merge(nomes, on="nome_estabelecimento", how="left")
    a["cidade_bloco"] = a.cidade.map(normalizar_cidade)

    excluidas = a[(a.nome_norm == "") | (a.cidade_bloco == "")]
    print(f"transações válidas: {len(a):,}; descartadas (app puro/sem nome/sem cidade): {len(excluidas):,}")
    print(excluidas.intermediador.fillna("sem nome/cidade").value_counts().to_string())
    a = a[(a.nome_norm != "") & (a.cidade_bloco != "")]

    ag = (
        a.groupby(["nome_norm", "cidade_bloco"])
        .agg(
            n_transacoes=("valor_lit", "size"),
            valor_total=("valor_lit", lambda v: v.sum() / 100),
            portadores=("conta_token", "nunique"),
            meses_ativos=("data_compra", lambda d: d.dt.month.nunique()),
            mcc=("mcc", lambda m: m.mode().iat[0] if m.notna().any() else None),
            via_app_entrega=("intermediador", lambda s: (s == "entrega").any()),
            exemplo_nome=("nome_estabelecimento", lambda s: s.value_counts().index[0]),
        )
        .reset_index()
    )
    ag["id_aberto"] = np.arange(len(ag))
    ag["truncado"] = ag.exemplo_nome.str.len() >= TAM_NOME_ABERTO - 1
    nome_base = ag.exemplo_nome.map(_base)
    ag["cidade_facilitador"] = nome_base.str.contains(_RE_QR) | (
        nome_base.str.contains(_RE_MERCADO_PAGO) & (ag.cidade_bloco == _CIDADE_SEDE_MERCADO_PAGO))
    ag["possivel_mei"] = ag.exemplo_nome.str.contains(_RE_CNPJ_RAIZ.pattern + r"\D", regex=True)
    ag["nucleo"] = ag.nome_norm.map(nucleo)
    return ag


def preparar_fechado():
    f = pd.read_parquet(DIR_DADOS / "arranjo_fechado_estabelecimentos.parquet")
    f["cidade_bloco"] = f.cidade.map(normalizar_cidade)
    variantes = []
    for col in ["nome_estabelecimento", "fantasia_estabelecimento", "nome_fantasia_app"]:
        v = f[["estabelecimento_token", "cidade_bloco", "situacao", col]].rename(columns={col: "nome_orig"})
        v["origem"] = col
        variantes.append(v)
        # versão truncada como no arranjo aberto
        t = v.copy()
        t["nome_orig"] = t.nome_orig.str[:TAM_NOME_ABERTO]
        t["origem"] = col + "_22"
        variantes.append(t)
    v = pd.concat(variantes, ignore_index=True)
    v["nome_norm"] = v.nome_orig.map(normalizar_nome)
    v = v[(v.nome_norm != "") & (v.cidade_bloco != "")]
    v["nucleo"] = v.nome_norm.map(nucleo)
    return v.drop_duplicates(["estabelecimento_token", "nome_norm"]).reset_index(drop=True)


def _top_k_por_cidade(ab, fe, col):
    """Retorna os K vizinhos mais próximos por TF-IDF de n-gramas dentro de cada cidade."""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)
    vec.fit(pd.concat([ab[col], fe[col]]).unique())
    Xa = vec.transform(ab[col])
    Xf = vec.transform(fe[col])

    idx_f_por_cidade = fe.groupby("cidade_bloco").indices
    sem_cidade = ab.cidade_facilitador.to_numpy()
    blocos = {c: ia[~sem_cidade[ia]] for c, ia in ab.groupby("cidade_bloco").indices.items()}
    # sem a cidade da loja, compara com o arranjo fechado inteiro
    if sem_cidade.any():
        idx_f_por_cidade["*"] = np.arange(len(fe))
        blocos["*"] = np.flatnonzero(sem_cidade)
    pares = []
    for cidade, ia in blocos.items():
        jf = idx_f_por_cidade.get(cidade)
        if jf is None or len(ia) == 0:
            continue
        Bf = Xf[jf].T.tocsc()
        for ini in range(0, len(ia), 2000):
            lote = ia[ini : ini + 2000]
            sim = (Xa[lote] @ Bf).toarray()
            k = min(TOP_K, sim.shape[1])
            top = np.argpartition(-sim, k - 1, axis=1)[:, :k]
            for linha, cols in enumerate(top):
                for c in cols:
                    if sim[linha, c] > 0:
                        pares.append((lote[linha], jf[c]))
    return pares


def gerar_candidatos(ab, fe):
    # candidatos pelo nome completo e pelo núcleo
    pares = _top_k_por_cidade(ab, fe, "nome_norm") + _top_k_por_cidade(ab, fe, "nucleo")
    p = pd.DataFrame(pares, columns=["i_aberto", "i_fechado"]).drop_duplicates()
    sem_cidade = (~ab.cidade_bloco.isin(set(fe.cidade_bloco))).sum()
    print(f"pares candidatos: {len(p):,}; estabelecimentos do aberto sem cidade no fechado: {sem_cidade:,}")
    return p


def _similaridades(a, b):
    return JaroWinkler.similarity(a, b), fuzz.token_sort_ratio(a, b) / 100


def pontuar(ab, fe, p):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)
    vec.fit(pd.concat([ab.nome_norm, fe.nome_norm]).unique())
    Xa = vec.transform(ab.nome_norm.to_numpy()[p.i_aberto])
    Xf = vec.transform(fe.nome_norm.to_numpy()[p.i_fechado])
    p["tfidf"] = np.asarray(Xa.multiply(Xf).sum(axis=1)).ravel()

    na = ab.nome_norm.to_numpy()[p.i_aberto]
    nf = fe.nome_norm.to_numpy()[p.i_fechado]
    sims = [_similaridades(x, y) for x, y in zip(na, nf)]
    p["jaro_winkler"] = [s[0] for s in sims]
    p["token_sort"] = [s[1] for s in sims]
    p["token_set"] = [fuzz.token_set_ratio(x, y) / 100 for x, y in zip(na, nf)]
    # token set fica fora do score porque gera falso positivo quando um nome contém o outro
    p["score_nome"] = p[["tfidf", "jaro_winkler", "token_sort"]].mean(axis=1)

    # se o nome foi truncado, compara com o prefixo do núcleo do fechado
    ua = ab.nucleo.to_numpy()[p.i_aberto]
    uf = fe.nucleo.to_numpy()[p.i_fechado]
    trunc = ab.truncado.to_numpy()[p.i_aberto]
    score_nuc = []
    for x, y, t in zip(ua, uf, trunc):
        if len(x) < 3 or len(y) < 3:
            score_nuc.append(np.nan)
            continue
        if t and len(y) > len(x):
            y = y[: len(x)]
        score_nuc.append(np.mean(_similaridades(x, y)))
    p["score_nucleo"] = score_nuc
    # o núcleo entra em média com o nome completo e nunca reduz o score
    p["score"] = np.fmax(p.score_nome, (p.score_nome + p.score_nucleo) / 2)

    p = p.sort_values("score", ascending=False).drop_duplicates("i_aberto")
    p["nome_fechado"] = fe.nome_orig.to_numpy()[p.i_fechado]
    p["nome_norm_fechado"] = fe.nome_norm.to_numpy()[p.i_fechado]
    p["token_fechado"] = fe.estabelecimento_token.to_numpy()[p.i_fechado]
    p["situacao_fechado"] = fe.situacao.to_numpy()[p.i_fechado]

    r = ab.merge(p.drop(columns="i_fechado"), left_on="id_aberto", right_on="i_aberto", how="left")
    r["score"] = r.score.fillna(0)
    # nome só com termos genéricos nunca é declarado credenciado
    nucleo_fechado = r.nome_norm_fechado.fillna("").map(nucleo)
    generico = (r.nucleo.str.len() < 2) | (nucleo_fechado.str.len() < 2)
    # mesma marca indica a rede, não necessariamente a mesma unidade
    marca_a = r.nucleo.str.split().str[0]
    marca_f = nucleo_fechado.str.split().str[0]
    r["mesma_marca"] = (marca_a == marca_f) & (marca_a.str.len() >= 4)
    r["faixa"] = np.select(
        # sem a cidade da loja o resultado fica no máximo em revisão
        [(r.score >= LIMIAR_CREDENCIADO) & ~generico & ~r.cidade_facilitador,
         r.score >= LIMIAR_REVISAO],
        ["credenciado", "revisao"],
        "nao_credenciado",
    )
    return r.drop(columns="i_aberto")


# termos que passam no critério de rede mas descrevem categoria, não marca
_MARCAS_DESCRITIVAS = {"BURGER", "LIVRARIA", "FARMACIAS", "POSTOS", "SUPERMARKET", "CASAS", "LOJAS"}

def marcas_de_rede(fe):
    """Retorna as marcas de rede (muitas unidades, poucas razões sociais) e os termos comuns."""
    v = fe[~fe.origem.str.startswith("nome_estabelecimento")].copy()
    v["marca"] = v.nucleo.str.split().str[0]
    v = v[v.marca.str.len() >= 3].drop_duplicates(["estabelecimento_token", "marca"])
    razao = fe[fe.origem == "nome_estabelecimento"].drop_duplicates("estabelecimento_token")
    v = v.merge(razao[["estabelecimento_token", "nome_norm"]].rename(columns={"nome_norm": "razao"}),
                on="estabelecimento_token")
    g = v.groupby("marca").agg(unidades=("estabelecimento_token", "nunique"), razoes=("razao", "nunique"))
    proporcao = g.razoes / g.unidades
    redes = set(g[(g.unidades >= 3) & (proporcao <= RAZOES_POR_UNIDADE_REDE)].index)
    # termos que iniciam nomes de muitas empresas diferentes, como prenomes
    comuns = set(g[(g.razoes >= 20) & (proporcao > RAZOES_POR_UNIDADE_REDE)].index)
    return redes, comuns


def classificar_listas(r, marcas_rede, termos_comuns):
    """Separa os alvos em rede credenciada, rede não credenciada, baixa abrangência e estabelecimento."""
    alvo = r[r.faixa != "credenciado"].copy()
    termos = alvo.nucleo.str.split()
    marca = termos.str[0]
    # nome de pessoa ou de MEI em várias cidades não é rede
    nome_de_pessoa = alvo.possivel_mei | ((termos.str.len() == 1) & marca.isin(termos_comuns))
    # cidade do facilitador não conta como cidade da rede
    validos = alvo[(alvo.nucleo.str.len() >= 2) & ~nome_de_pessoa & ~alvo.cidade_facilitador]

    # rede credenciada: mesma marca do candidato, marca de rede ou nome credenciado em outra cidade
    marca_rede = marca.isin(marcas_rede - _MARCAS_DESCRITIVAS)
    nomes_credenciados = set(r.loc[r.faixa == "credenciado", "nome_norm"])
    rede_credenciada = (
        (alvo.mesma_marca & marca.isin(marcas_rede))
        | (marca_rede & (marca.str.len() >= 6))
        | (alvo.index.isin(validos.index) & alvo.nome_norm.isin(nomes_credenciados))
    )
    cidades_por_nome = validos.groupby("nome_norm").cidade_bloco.nunique()
    multi_cidade = alvo.nome_norm.map(cidades_por_nome).fillna(0) >= 2

    alvo["lista"] = np.select(
        [rede_credenciada, multi_cidade, alvo.portadores < MIN_PORTADORES],
        ["rede_credenciada", "rede_nao_credenciada", "baixa_abrangencia"],
        "estabelecimento",
    )
    alvo["marca"] = np.where((alvo.lista == "rede_credenciada") & marca.isin(marcas_rede),
                             marca, alvo.nome_norm)

    # gasto alto em poucos portadores pode indicar uso indevido do benefício
    valor_por_portador = r.valor_total / r.portadores
    p99 = valor_por_portador.quantile(0.99)
    alvo["concentracao_atipica"] = (alvo.portadores < MIN_PORTADORES) & (
        alvo.valor_total / alvo.portadores >= p99)
    print(f"concentração atípica: valor por portador >= R$ {p99:,.2f} (P99)")
    return alvo


def priorizar(df):
    """Ordena pela média dos percentis de valor e de transações."""
    df = df.copy()
    df["pct_valor"] = df.valor_total.rank(pct=True)
    df["pct_transacoes"] = df.n_transacoes.rank(pct=True)
    df["indice_prioridade"] = (df.pct_valor + df.pct_transacoes) / 2
    df = df.sort_values(["indice_prioridade", "valor_total"], ascending=False)
    df["ranking"] = np.arange(1, len(df) + 1)
    return df


def agregar_redes(alvo):
    redes = alvo[alvo.lista.str.startswith("rede")]
    ag = (
        redes.groupby(["lista", "marca"])
        .agg(
            unidades=("id_aberto", "size"),
            cidades=("cidade_bloco", "nunique"),
            n_transacoes=("n_transacoes", "sum"),
            valor_total=("valor_total", "sum"),
            portadores=("portadores", "sum"),  # soma por unidade, pode repetir pessoas
            lista_cidades=("cidade_bloco", lambda c: ", ".join(sorted(set(c))[:10])),
            exemplo_nome=("exemplo_nome", "first"),
        )
        .reset_index()
    )
    return pd.concat([priorizar(g) for _, g in ag.groupby("lista")], ignore_index=True)


def amostra_rotulagem(r, n_por_faixa=150, semente=42):
    """Amostra estratificada por faixa de score para montar o gabarito manual."""
    r = r.assign(decil=pd.cut(r.score, [0, .5, .6, .7, .75, .8, .85, .9, .95, 1.01], right=False))
    am = (
        r[r.score > 0]
        .sample(frac=1, random_state=semente)
        .groupby("decil", observed=True)
        .head(n_por_faixa // 3)
        .sort_values("score")
    )
    cols = ["id_aberto", "decil", "score", "exemplo_nome", "nome_norm", "cidade_bloco",
            "nome_fechado", "nome_norm_fechado", "tfidf", "jaro_winkler", "token_sort", "token_set",
            "score_nome", "score_nucleo"]
    return am[cols].assign(eh_mesmo_estabelecimento="")


def main():
    DIR_SAIDA.mkdir(exist_ok=True)
    ab = preparar_aberto()
    fe = preparar_fechado()
    ab = reconciliar_cidades(ab, fe)
    print(f"estabelecimentos aberto (nome+cidade): {len(ab):,}; variantes de nome no fechado: {len(fe):,}")
    p = gerar_candidatos(ab, fe)
    r = pontuar(ab, fe, p)

    resumo = r.groupby("faixa").agg(estabelecimentos=("id_aberto", "size"),
                                    transacoes=("n_transacoes", "sum"),
                                    valor=("valor_total", "sum"))
    resumo["pct_valor"] = resumo.valor / resumo.valor.sum()
    print(resumo.to_string())

    alvo = classificar_listas(r, *marcas_de_rede(fe))
    r.to_parquet(DIR_SAIDA / "match_completo.parquet", index=False)
    amostra_rotulagem(r).to_csv(DIR_SAIDA / "amostra_rotulagem.csv", index=False)

    resumo = alvo.groupby("lista").agg(estabelecimentos=("id_aberto", "size"),
                                       transacoes=("n_transacoes", "sum"),
                                       valor=("valor_total", "sum"),
                                       concentracao_atipica=("concentracao_atipica", "sum"))
    resumo["pct_valor"] = resumo.valor / resumo.valor.sum()
    print(resumo.to_string())

    cols = ["ranking", "exemplo_nome", "cidade_bloco", "mcc", "n_transacoes", "valor_total",
            "portadores", "faixa", "score", "nome_fechado", "possivel_mei"]
    est = priorizar(alvo[alvo.lista == "estabelecimento"])
    est.to_csv(DIR_SAIDA / "ranking_estabelecimentos.csv", index=False)
    print(est[cols].head(25).to_string(index=False))

    redes = agregar_redes(alvo)
    redes.to_csv(DIR_SAIDA / "ranking_redes.csv", index=False)
    alvo[alvo.lista.str.startswith("rede")].to_csv(DIR_SAIDA / "redes_unidades.csv", index=False)
    for lista, g in redes.groupby("lista"):
        print(f"--- {lista}")
        print(g.drop(columns=["lista", "lista_cidades"]).head(12).to_string(index=False))

    baixa = priorizar(alvo[alvo.lista == "baixa_abrangencia"])
    baixa.to_csv(DIR_SAIDA / "baixa_abrangencia.csv", index=False)
    print("--- concentração atípica (maiores valores)")
    print(baixa[baixa.concentracao_atipica].nlargest(10, "valor_total")[cols[1:]].to_string(index=False))


if __name__ == "__main__":
    main()
