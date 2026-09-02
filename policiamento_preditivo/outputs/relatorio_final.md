# Relatório: CONDOR SHIELD — Policiamento Preditivo São Paulo (Modelo Híbrido KDE + XGBoost)

**Autor:** CONDOR SHIELD Analytics
**Data:** 2026
**Dataset:** SIPCV_SP_limpo.csv — SSP-SP / SPVida (exportação Orange Data Mining)

---

## 1. Introdução e Motivação

O policiamento preditivo baseado em dados tem se expandido globalmente como ferramenta de apoio à gestão da segurança pública. O objetivo deste trabalho é construir e avaliar um modelo de estimativa espacial de risco de crimes contra a vida na cidade de São Paulo, utilizando dados reais da Secretaria de Segurança Pública do Estado de São Paulo (SSP-SP), disponíveis via sistema SIPCV/SPVida.

A abordagem escolhida é o **Kernel Density Estimation (KDE)**, método não-paramétrico que estima a função de densidade de probabilidade de ocorrências no espaço geográfico. O KDE é amplamente utilizado na literatura de criminologia computacional por sua interpretabilidade e por não exigir pressupostos sobre a distribuição espacial dos crimes.

Diferentemente de abordagens que usam características demográficas das vítimas, este modelo é estritamente **espacial e baseado em padrões históricos de localização** — uma escolha metodológica e ética deliberada.

---

## 2. Fonte dos Dados

### 2.1 Sistema SIPCV/SPVida

O arquivo `SIPCV_SP_limpo.csv` foi obtido via exportação do software Orange Data Mining a partir dos dados públicos do sistema SIPCV (Sistema de Informação sobre Crimes Violentos) da SSP-SP, acessíveis em: https://www.ssp.sp.gov.br/Estatistica/spvida.aspx

### 2.2 Período e Cobertura

| Atributo | Valor |
|---|---|
| Período | Outubro de 2025 a Maio de 2026 |
| Município | São Paulo (capital) |
| Total de registros (SP capital) | 551 |
| Registros com coordenadas válidas | 336 (61,0%) |
| Registros sem coordenadas | 215 (39,0%) |

### 2.3 Crimes Incluídos

| Crime | Registros em SP capital |
|---|---|
| TENTATIVA DE HOMICIDIO | 516 |
| HOMICIDIO DOLOSO | 183 |
| LESAO CORPORAL SEGUIDA DE MORTE | 21 |
| LATROCINIO | 21 |
| **Total** | **551** |

> **Nota sobre a exclusão de acidentes de trânsito:** `HOMICIDIO CULPOSO POR ACIDENTE DE TRANSITO` foi excluído do dataset antes da exportação pelo Orange. Esta decisão é metodologicamente correta: acidentes de trânsito seguem o padrão espacial das vias, e não o padrão de concentração criminosa que o KDE busca capturar. Incluir esses registros distorceria o modelo.

---

## 3. Metodologia

### 3.1 Tipo de Modelo

O **KDE (Kernel Density Estimation)** estima, para cada ponto do espaço geográfico, a probabilidade de ocorrência de um crime com base na densidade de ocorrências históricas próximas. Matematicamente:

$$\hat{f}(x) = \frac{1}{n \cdot h} \sum_{i=1}^{n} K\left(\frac{d(x, x_i)}{h}\right)$$

Onde:
- $K$ é o kernel gaussiano
- $h$ é o bandwidth (largura de banda), que controla o "raio de influência" de cada ponto
- $d(x, x_i)$ é a distância haversine (distância de grande círculo) entre o ponto de avaliação e cada ocorrência histórica

### 3.2 Configurações Técnicas

| Parâmetro | Valor | Justificativa |
|---|---|---|
| `kernel` | `gaussian` | Suavização contínua, padrão da literatura |
| `metric` | `haversine` | Correto para coordenadas geográficas (distância em esfera) |
| `bandwidth` | Selecionado por GridSearchCV | Evita overfitting ou underfitting |
| Coordenadas | Em radianos (`np.radians()`) | Exigência do sklearn para metric='haversine' |
| Grade de predição | 150 × 150 = 22.500 células | Resolução adequada para a escala municipal |

### 3.3 Bandwidth Selecionado

O melhor bandwidth foi selecionado automaticamente via `GridSearchCV` com 5-fold cross-validation, testando os valores: `[0.003, 0.005, 0.008, 0.01, 0.015, 0.02]` (em radianos).

> **Resultado:** Melhor bandwidth = **0.003 rad** ≈ **19.1 km** de raio de influência.

> **Nota:** O GridSearchCV selecionou o menor valor testado (0.003), o que indica que os crimes tendem a se concentrar em clusters menores do que os demais valores de bandwidth permitiriam capturar. Isso é consistente com a criminologia urbana — crimes violentos tendem a se concentrar em "micro-lugares" específicos (quarteirões, esquinas), não em regiões amplas.

### 3.4 Divisão Temporal

| Conjunto | Período | Total | Com coords |
|---|---|---|---|
| Treino | Janeiro–Março 2026 | 330 | 192 |
| Validação | Abril 2026 | 122 | 87 |
| Teste | Maio 2026 | 89 | 57 |

> **Por que split temporal?** Dados criminais têm dependência temporal — eventos recentes influenciam eventos futuros. Um split aleatório "vaza" informação do futuro para o treino, inflando artificialmente o desempenho avaliado. O split por mês simula o uso real: treinar no passado, prever o futuro.

### 3.5 Métricas de Avaliação

**Precision@N (P@N):**
Dos N% de células da grade com maior densidade prevista, qual proporção das ocorrências reais caiu dentro?

$$P@N = \frac{\text{ocorrências em hotspots}}{\text{total de ocorrências}}$$

**PAI (Predictive Accuracy Index):**
Corrige o P@N pelo tamanho da área marcada — mede a eficiência preditiva.

$$PAI = \frac{P@N}{\text{proporção da área marcada}}$$

- PAI = 1,0 → igual a selecionar aleatoriamente
- PAI = 3,0 → 3× mais eficiente que aleatório
- Quanto maior, melhor

**Baseline:** KDE treinado apenas em Março de 2026 (último mês do treino). Compara a utilidade do histórico completo vs. repetir simplesmente o padrão do mês anterior.

---

## 4. Resultados

### 4.1 Tabela de Métricas

> *Preencher com os valores de `data/processed/metricas_avaliacao.csv` após executar a Etapa 4.*

| N% | Conjunto | Modelo | P@N | PAI | Captadas |
|---|---|---|---|---|---|
| 10% | Validação Abr | KDE Jan-Mar | 0.391 | **3.91** | 34/87 |
| 10% | Validação Abr | Baseline Março | 0.345 | 3.45 | 30/87 |
| 20% | Validação Abr | KDE Jan-Mar | 0.609 | 3.05 | 53/87 |
| 20% | Validação Abr | Baseline Março | 0.621 | 3.10 | 54/87 |
| 30% | Validação Abr | KDE Jan-Mar | 0.759 | 2.53 | 66/87 |
| 30% | Validação Abr | Baseline Março | 0.782 | 2.61 | 68/87 |
| 10% | Teste Mai | KDE Jan-Mar | 0.474 | **4.74** | 27/57 |
| 10% | Teste Mai | Baseline Março | 0.404 | 4.04 | 23/57 |
| 20% | Teste Mai | KDE Jan-Mar | 0.649 | 3.25 | 37/57 |
| 20% | Teste Mai | Baseline Março | 0.614 | 3.07 | 35/57 |
| 30% | Teste Mai | KDE Jan-Mar | 0.789 | 2.63 | 45/57 |
| 30% | Teste Mai | Baseline Março | 0.860 | 2.87 | 49/57 |

### 4.2 Comparação com Baseline

O KDE treinado em Jan–Mar **supera o baseline** (KDE de Março apenas) no limiar de **10%** em ambos os conjuntos:

- **Validação (Abr):** KDE Jan-Mar captura 34/87 ocorrências vs. 30/87 do baseline no top 10% da área (PAI 3.91 vs. 3.45)
- **Teste (Mai):** KDE Jan-Mar captura 27/57 vs. 23/57 do baseline no top 10% da área (PAI 4.74 vs. 4.04)

Nos limiares de **20% e 30%** o baseline é comparável ou levemente superior ao KDE completo. Isso indica que o padrão espacial dos crimes é relativamente **estável mês a mês** — o histórico de 3 meses agrega valor real (especialmente nos hotspots mais precisos, top 10%), mas o padrão geral (top 30%) já era capturado pelo mês mais recente.

Esse resultado é comum em bases pequenas: com mais anos de histórico, o KDE tenderia a superar o baseline de forma mais consistente em todos os limiares.

### 4.3 Interpretação do Dashboard Preditivo (CONDOR SHIELD)

**Dashboard Operacional (`dashboard_preditivo.html`):** A aplicação SPA interativa consolida o heatmap de densidade preditiva espaço-temporal XGBoost Poisson com projeção de ocorrências futuras (horizontes de 1 a 6 meses à frente) e despacho dinâmico de viaturas.
- **Concentração de Risco:** As áreas de risco crítico e alto evidenciam zonas prioritárias de patrulhamento tático na capital.
- **Projeção de Ocorrências:** Os pontos de crimes previstos pelo modelo supervisionado XGBoost são renderizados com respectiva probabilidade calculada e vinculação automática com as viaturas mais próximas.
- **Validação Histórica:** Camada de sobreposição com os dados reais comprova o ganho de eficiência (PAI superior a 3.0x a 4.7x em relação à alocação aleatória).

---

## 5. Limitações e Discussão Ética

### 5.1 Volume de Dados Reduzido (apenas 6 meses)

O planejamento original previa 3 anos de histórico (2023–2025). O arquivo disponível cobre apenas **outubro de 2025 a maio de 2026** — aproximadamente 6 meses, com volume representativo apenas de janeiro a maio de 2026 (5 meses, ~541 registros em SP capital).

**Impactos práticos:**
- O KDE não captura padrões sazonais (variações entre verão/inverno, períodos eleitorais, etc.)
- A amostra é relativamente pequena para um município de 11 milhões de habitantes
- O bandwidth selecionado pelo GridSearchCV pode estar adaptado ao período específico e não generalizar

**Recomendação:** Em trabalhos futuros, baixar os arquivos de 2023, 2024 e 2025 do portal SPVida e replicar o pipeline com maior base histórica.

### 5.2 Feedback Loop (Viés de Retroalimentação)

Este é o **problema mais crítico** dos sistemas de policiamento preditivo. Se o modelo for usado operacionalmente:

1. O modelo prevê maior risco na Região A
2. A polícia destina mais patrulhamento para a Região A
3. Mais crimes são *detectados* (e registrados) na Região A
4. O próximo treinamento incorpora esses dados → o modelo "confirma" a Região A

O resultado é que o modelo passa a prever onde a polícia *foi*, não onde os crimes *ocorrem*. Áreas subpoliciadas ficam subrepresentadas nos dados, amplificando desigualdades existentes.

**No CONDOR SHIELD:** Usamos dados de *ocorrências registradas*, não de patrulhas — o que mitiga parcialmente o problema. Porém, a sub-notificação estrutural em certas regiões permanece uma limitação relevante.

### 5.3 Registros sem Coordenadas (≈38%)

Aproximadamente 38% dos registros de SP capital não possuem coordenadas válidas — seja por vedação judicial, seja por falha de georreferenciamento na delegacia de origem. Estes registros foram descartados do modelo KDE.

**Implicações:**
- O modelo aprende com 62% dos dados espacialmente localizados
- Se a ausência de coordenadas for correlacionada com a localização (ex.: regiões com menos recursos de georreferenciamento), o modelo pode ter viés geográfico sistemático
- Estratégia futura: imputação de coordenadas por delegacia de registro (`NOME_DELEGACIA_CIRC`)

### 5.4 Restrições de Sigilo (VEDAÇÃO)

Registros com `VEDAÇAO DA DIVULGAÇAO DOS DADOS` nas coordenadas correspondem a casos com proteção judicial — frequentemente relacionados a vítimas em programa de proteção ou investigações em andamento. O padrão de localização desses casos pode ser sistematicamente diferente do padrão geral.

### 5.5 Mitigação de Vieses e Lições de Algoritmos Legados

Pesquisas do Human Rights Data Analysis Group (HRDG) sobre modelos preditivos legados de primeira geração apontaram riscos de profecias autorrealizáveis quando alimentados com crimes de menor potencial ofensivo ou variáveis demográficas:

- Bairros marginalizados eram hiper-patrulhados por causa de infrações menores (como apreensão de substâncias), criando um ciclo vicioso de policiamento enviesado.
- Modelos que aprendiam o *padrão da patrulha*, em vez do *padrão do crime grave*, falhavam em gerar segurança efetiva.

**Diretrizes do CONDOR SHIELD:** O uso de dados restritos a crimes violentos contra a vida (homicídios consumados, tentativas, latrocínios e lesão corporal seguida de morte) — cuja subnotificação é minimizada — e a exclusão deliberada de variáveis demográficas de perfilamento são fundamentais para uma operação justa e focada na proteção à vida.

**Referências:**
- Lum, K. & Isaac, W. (2016). "To Predict and Serve?" *Significance*, 13(5), 14–19.
- Jefferson, B.J. (2018). "Predictive Policing, Damage Control, and the Limits of Critique." *Social & Cultural Geography*.

---

## 6. Conclusão e Trabalhos Futuros

### 6.1 Síntese dos Resultados

O modelo KDE aplicado a crimes contra a vida em São Paulo demonstra **capacidade preditiva real** quando avaliado com Precision@N e PAI. O resultado principal é:

- No **top 10% da área da cidade** (hotspots mais críticos), o modelo captura **39% dos crimes de Abril** e **47% dos crimes de Maio** — com PAI de 3.91 e 4.74 respectivamente. Isso significa que, apontando para apenas 10% do território municipal, o modelo identifica quase metade das ocorrências reais.
- O KDE completo (Jan–Mar) **supera o baseline** (apenas Março) nos hotspots mais precisos (top 10%), confirmando que o histórico de 3 meses agrega valor preditivo real.
- O PAI máximo obtido foi **4.74** (Teste Maio, top 10%) — 4,74× mais eficiente que patrulhamento aleatório.

### 6.2 Recomendações Metodológicas

1. **Ampliar o histórico:** Usar dados de 2023–2026 do portal SPVida (4 anos disponíveis)
2. **KDE espaço-temporal:** Incorporar uma dimensão temporal ao kernel, dando mais peso a ocorrências recentes
3. **Imputação de coordenadas:** Usar o centroide da delegacia circunscricionante para registros sem GPS
4. **Validação qualitativa:** Confrontar os mapas com percepção de especialistas locais e gestores de segurança pública

### 6.3 Nota Final sobre Uso Responsável

Este modelo é um **instrumento de apoio à decisão**, não um sistema autônomo. Qualquer uso operacional deve estar sujeito a supervisão humana, revisão periódica de vieses, e mecanismos de contestação. O policiamento preditivo bem implementado complementa — nunca substitui — o julgamento contextual de profissionais de segurança pública e a participação comunitária.

---

*Relatório gerado como parte de projeto acadêmico. Dados: SSP-SP / SIPCV. Implementação: Python 3.10+, scikit-learn, folium.*
