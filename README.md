# record-linkage-estabelecimentos

Código do Trabalho de Conclusão de Curso **"Identificação de estabelecimentos prioritários para credenciamento por processamento de linguagem natural"**, do MBA em Data Science e Analytics da USP/Esalq.

Autora: Larissa dos Santos Magistrali. Orientador: Edilson José Rodrigues.

## O que o código faz

Uma operadora de cartões de benefícios registra transações em estabelecimentos fora da sua rede credenciada, mas as duas bases não têm chave comum. O nome do estabelecimento chega truncado nas transações, sem documento de identificação. O código casa esses nomes com o cadastro da rede credenciada ("record linkage"). Em seguida, identifica os estabelecimentos ainda não credenciados e os ordena pela demanda observada.

Etapas:

1. **Preparação.** Seleciona as compras aprovadas e trata os aplicativos de entrega e de mobilidade. Normaliza os nomes: remove acentos, prefixos de facilitadores, raiz de CNPJ, numeração de loja e sufixos societários.
2. **Blocagem e candidatos.** Compara só estabelecimentos da mesma cidade e seleciona candidatos por TF-IDF de n-gramas de caracteres.
3. **Escore e decisão.** Combina cosseno TF-IDF, Jaro-Winkler e "token sort ratio". Classifica cada estabelecimento como credenciado, revisão ou não credenciado.
4. **Avaliação.** Mede o erro numa amostra rotulada às cegas e estratificada por escore, com pesos por estrato e intervalos de Wilson.
5. **Comparação de métodos.** Compara o escore com regressão logística e com embeddings de sentenças, usando precisão média e intervalo por bootstrap pareado.
6. **Priorização.** Separa os não credenciados em listas por ação comercial: unidades de redes credenciadas, redes não credenciadas, baixa abrangência e estabelecimentos individuais. Ordena cada lista por um índice que combina valor e quantidade de transações.

## Dados

Os dados são restritos e **não fazem parte deste repositório**. O código pode ser aplicado a outras bases com a mesma estrutura.

| Arquivo esperado em`dataset/`                       | Colunas usadas                                                                                                                                          |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `arranjo_aberto_transacoes_2026_01_2026_06.parquet` | `tipo_evento`, `tipo_transacao`, `aprovada`, `cancelada`, `data_compra`, `valor_lit` (centavos), `nome_estabelecimento`, `cidade`, `mcc`, `conta_token` |
| `arranjo_fechado_estabelecimentos.parquet`          | `estabelecimento_token`, `nome_estabelecimento`, `fantasia_estabelecimento`, `nome_fantasia_app`, `cidade`, `situacao`                                  |

## Estrutura

```
codigo/
  pipeline_match.py       pareamento, faixas de decisão e listas de priorização
  avaliacao.py            métricas na amostra rotulada
  comparacao_modelos.py   comparação de métodos de similaridade
  tabelas_graficos.py     tabelas (Word) e gráficos (Excel e PNG) do texto
```

As pastas `dataset/`, `rotulagem/` e `resultados/` ficam no nível acima de `codigo/` e não são versionadas.

## Como executar

Python 3.11. Instale as dependências:

```
pip install -r requirements.txt
```

Execute os programas nesta ordem, a partir da pasta `codigo/`:

```
python pipeline_match.py
python avaliacao.py
python comparacao_modelos.py
python tabelas_graficos.py
```

A avaliação confere se os pares da amostra rotulada coincidem com os da execução corrente e para em caso de divergência. Se o ambiente tiver TensorFlow com Keras 3 instalado, rode a comparação com `USE_TF=0`.

## Privacidade

Os identificadores de empresa, conta, cartão e estabelecimento foram pseudonimizados com HMAC-SHA256 na extração. Nenhum nome de estabelecimento, pessoa ou empresa é publicado neste repositório.
