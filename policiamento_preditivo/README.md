# CONDOR SHIELD — Policiamento Preditivo (São Paulo)

Sistema de **Inteligência Policial e Policiamento Preditivo Espaço-Temporal (XGBoost Poisson)** aplicado à estimativa de risco de crimes contra a vida na cidade de São Paulo, projeção de ocorrências futuras e direcionamento operacional de viaturas com base em dados reais da SSP-SP (SIPCV/SPVida).

## Estrutura do Projeto

```
policiamento_preditivo/
├── data/
│   ├── raw/
│   │   └── SIPCV_SP_limpo.csv        ← dataset de entrada (exportação Orange)
│   └── processed/
│       ├── df_sp_limpo.csv           ← SP capital, colunas selecionadas
│       ├── df_treino.csv             ← Jan–Mar 2026 (330 registros)
│       ├── df_validacao.csv          ← Abril 2026 (122 registros)
│       ├── df_teste.csv              ← Maio 2026 (89 registros)
│       ├── grade_densidade.csv       ← grade 150×150 com scores KDE
│       ├── metricas_avaliacao.csv    ← P@N e PAI para todos os limiares (KDE)
│       ├── painel_delegacia_semana.csv ← painel espaço-temporal (Etapa 7)
│       ├── metricas_hibrido.csv      ← métricas comparativas KDE vs XGBoost (Etapa 9)
│       └── previsoes_xgboost_horizonte.json ← previsões e pontos de crimes previstos (Etapa 10)
├── models/
│   ├── kde_model.pkl                 ← modelo KDE serializado (prior espacial)
│   └── xgboost_model.pkl             ← modelo XGBoost Poisson (Etapa 8)
├── outputs/
│   ├── dashboard_preditivo.html      ← ★ CONDOR SHIELD: SPA interativo com pontos previstos & viaturas
│   └── relatorio_final.md            ← relatório com métricas
├── src/
│   ├── 01_limpeza.py
│   ├── 02_engenharia_features.py
│   ├── 03_treino_kde.py
│   ├── 04_avaliacao.py
│   ├── 05_visualizacao.py
│   ├── 06_dashboard.py               ← gerador do SPA interativo com direcionamento de viaturas
│   ├── 07_engenharia_temporal.py     ← painel espaço-temporal delegacia x semana
│   ├── 08_treino_xgboost.py          ← treino supervisionado XGBoost Poisson
│   ├── 09_avaliacao_hibrida.py       ← avaliação comparativa (KDE vs XGBoost)
│   └── 10_previsao_multi_horizonte.py ← rollout autorregressivo e pontos de crimes previstos
├── requirements.txt
└── README.md
```

## Pré-requisitos

- Python 3.10+
- O arquivo `SIPCV_SP_limpo.csv` deve estar em `data/raw/`

## Instalação

```bash
pip install -r requirements.txt
```

## Como Reproduzir do Zero

Execute os scripts **na ordem exata** a partir do diretório `policiamento_preditivo/`:

```bash
# Etapa 1: limpeza, filtro SP capital e split temporal (Jan–Mar / Abr / Mai)
python src/01_limpeza.py

# Etapa 2: criação de features (TURNO, FIM_DE_SEMANA, VIA_PUBLICA, CRIME_DOLOSO)
python src/02_engenharia_features.py

# Etapa 3: treino do KDE + grade de densidade 150×150 (1–2 min com GridSearchCV)
python src/03_treino_kde.py

# Etapa 4: avaliação (Precision@N e PAI vs. baseline de Março)
python src/04_avaliacao.py

# Etapa 5: geração dos mapas HTML interativos
python src/05_visualizacao.py

# Etapa 7: engenharia de features espaço-temporais (painel delegacia x semana)
python src/07_engenharia_temporal.py

# Etapa 8: treino do XGBoost com perda Poisson (regressão de contagem)
python src/08_treino_xgboost.py

# Etapa 9: avaliação comparativa (KDE isolado vs. XGBoost Híbrido)
python src/09_avaliacao_hibrida.py

# Etapa 10: rollout autorregressivo e geração de pontos de crimes previstos
python src/10_previsao_multi_horizonte.py

# Etapa 6: compilação do dashboard preditivo interativo CONDOR SHIELD
python src/06_dashboard.py
```

Após a execução, abra no navegador:

- `outputs/dashboard_preditivo.html` — **CONDOR SHIELD: Dashboard Preditivo XGBoost**
- `../index.html` — **Portal Central CONDOR SHIELD**

## Descrição do Dataset

| Propriedade | Valor |
|---|---|
| Fonte | SSP-SP via portal SPVida / Orange Data Mining |
| Arquivo | `SIPCV_SP_limpo.csv` |
| Encoding | UTF-8 com BOM (`utf-8-sig`) |
| Separador | vírgula (`,`) |
| Linhas brutas | 3.038 (inclui 2 linhas de metadados do Orange) |
| Linhas de dados | 3.036 |
| Colunas | 59 |
| Período | Outubro 2025 – Maio 2026 |

### Crimes selecionados (filtro Orange)

| Crime | Registros (SP capital) |
|---|---|
| TENTATIVA DE HOMICIDIO | 516 |
| HOMICIDIO DOLOSO | 183 |
| LESAO CORPORAL SEGUIDA DE MORTE | 21 |
| LATROCINIO | 21 |
| **TOTAL** | **551** |

> `HOMICIDIO CULPOSO POR ACIDENTE DE TRANSITO` foi **excluído** propositalmente — seu padrão espacial segue vias de tráfego, não concentração criminosa.

### Problemas de qualidade conhecidos

| Problema | Causa | Solução aplicada |
|---|---|---|
| Coordenadas com vírgula decimal | Exportação Orange (formato BR) | `.str.replace(',', '.')` antes de `pd.to_numeric()` |
| Texto de vedação no lugar de coords | Sigilo judicial | Substituição por `pd.NA` |
| Datas em DD/MM/YYYY | Padrão brasileiro | `pd.to_datetime(..., dayfirst=True)` |
| Variações sujas em texto | Inconsistência de entrada | Dicionários de mapeamento em `01_limpeza.py` |
| 2 linhas de metadados no CSV | Formato Orange | `skiprows=[1, 2]` + `header=0` |

## Metodologia (resumo)

1. **Modelo**: KDE espacial com `kernel='gaussian'` e `metric='haversine'`
2. **Bandwidth**: selecionado por `GridSearchCV` com cross-validation 5-fold
3. **Coordenadas**: convertidas para radianos antes do `.fit()` (requisito do haversine)
4. **Split**: por mês (não aleatório) — treino Jan–Mar, validação Abr, teste Mai
5. **Métricas**: Precision@N e PAI (Predictive Accuracy Index) em 3 limiares (10%, 20%, 30%)
6. **Baseline**: KDE treinado somente em Março

## Arquitetura do Modelo Híbrido (KDE + XGBoost Espaço-Temporal)

O modelo híbrido estende o modelo KDE integrando dinâmicas temporais e supervisionadas:
1. **Prior Espacial (KDE)**: O score de densidade estática obtido em `models/kde_model.pkl` para o centróide de cada delegacia serve como a feature `score_kde`.
2. **Painel Espaço-Temporal (Delegacia × Semana)**: Agrupa os dados de ocorrências por delegacia de circunscrição e semana (`data/processed/painel_delegacia_semana.csv`), garantindo a inclusão explícita de semanas sem ocorrências (contagem zero).
3. **Features Temporais e Lags**:
   - `lag_1_semana`, `lag_2_semanas`: Ocorrências nas semanas anteriores.
   - `media_4_semanas`, `media_8_semanas`: Média móvel de curto e médio prazo.
   - `tendencia`: Variação entre a última semana e a anterior (`lag_1 - lag_2`).
   - `MES`, `SEMANA_ANO`: Variáveis de calendário.
4. **Regressão de Contagem (XGBoost Poisson)**:
   - Objetivo: `count:poisson` com métrica `poisson-nloglik`.
   - Adequado para distribuições de contagem esparsas (~85-90% de zeros).
   - O modelo prediz a contagem esperada de ocorrências por delegacia/semana e é avaliado quanto ao PAI e Precision@N em relação ao KDE isolado.


## Decisões Éticas

As colunas `COR_CURTIS`, `SEXO_PESSOA`, `IDENTIDADE_GENERO`, `ORIENTACAO_SEXUAL` e `IDADE_DATA_OCORRENCIA` **não são usadas como features** do modelo. Usar características demográficas das vítimas como preditores é uma prática discriminatória documentada na literatura (ver relatório, Seção 5).

## Trabalhos Futuros

- Baixar dados de 2023, 2024 e 2025 completos do portal SPVida para ampliar o histórico de 6 meses para 3+ anos
- Incorporar dados contextuais (IDH por bairro, presença de equipamentos públicos) sem usar variáveis demográficas das vítimas
- Avaliar modelos alternativos: Prospective Hotspot Mapping, ST-KDE (espaço-temporal)
- Validação com especialistas em segurança pública (policiais, gestores, pesquisadores)

## Autor e Contexto

Projeto acadêmico — aluno iniciante em IA. Implementado com auxílio de documentação técnica detalhada. Prazo: 1 mês.
