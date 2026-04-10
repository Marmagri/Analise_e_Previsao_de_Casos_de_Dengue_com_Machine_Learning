# Análise_e_Previsão_de_Casos_de_Dengue_com_Machine_Learning
A análise foi conduzida na linguagem Python, utilizando bibliotecas como Pandas, RapidFuzz e Scikit-learn. Após a padronização dos dados, com destaque para a correção dos nomes de bairros via fuzzy matching, foi aplicado um modelo de aprendizado de máquina do tipo Random Forest Regressor. 

1. CONTEXTO E MOTIVAÇÃO
	O papel da Vigilância Epidemiológica é essencial na promoção e proteção da saúde pública, envolvendo o monitoramento de doenças e agravos à saúde, a coleta e análise de dados, e a implementação de ações preventivas e de controle (1).
As arboviroses são transmitidas pelo mosquito do gênero Aedes, especialmente, a espécie Aedes aegypti. Dentre elas, a dengue é a de maior relevância nas Américas, sendo considerada um dos principais problemas de saúde pública. O vírus da dengue (DENV) é um RNA-vírus pertencente ao gênero Flavivirus, da família Flaviviridae, e possui quatro sorotipos conhecidos (2).
Trata-se de uma doença febril aguda, sistêmica e dinâmica, com ampla variedade clínica, que vai desde casos assintomáticos até formas graves com risco de óbito (1). A notificação da dengue é compulsória, e os registros devem ser inseridos no Sistema de Informação de Agravos de Notificação (Sinan). A agilidade na coleta, análise e disseminação dos dados é crucial para a resposta oportuna dos órgãos de saúde, permitindo uma atuação integrada e eficaz (2).
Diante da situação epidemiológica, o Estado de São Paulo decretou emergência em saúde pública em 19/02/2025, e o município de Osasco adotou a mesma medida em 17/03/2025. Este trabalho foi motivado pela necessidade de direcionar as ações de controle de forma mais eficaz, em parceria com o Centro de Controle de Zoonoses.


1.1 Objetivo
	Os objetivos deste trabalho são:
Conhecer o padrão de distribuição espacial e temporal dos casos notificados de dengue em Osasco (majoritariamente autóctones — ou seja, com infecção contraída dentro da área estudada), com base nos bairros de residência;
Prever os bairros com maior risco de concentração de casos, a fim de subsidiar ações de prevenção e controle.


2. ATIVIDADES PRINCIPAIS
As atividades da Vigilância em Saúde envolvem a alimentação e análise contínua do Sinan, com foco no acompanhamento das arboviroses, construção de indicadores epidemiológicos, avaliação de programas e apoio a estudos e pesquisas. Entre seus objetivos está a identificação precoce de áreas de maior incidência, para orientar ações integradas de prevenção, controle e organização da assistência (1).
Com base na experiência prática da autora na Vigilância Epidemiológica e sua formação em andamento em Ciência de Dados pela Universidade Virtual do Estado de São Paulo, identificou-se a oportunidade de aplicar técnicas de ciência de dados na rotina de trabalho.

3.DESENVOLVIMENTO DO TRABALHO (METODOLOGIA E RESULTADOS)
	Este estudo foi conduzido com o uso da linguagem de programação Python, amplamente adotada na análise de dados e projetos de ciência de dados devido à sua sintaxe acessível e à robustez de bibliotecas especializadas, como Pandas, NumPy, Matplotlib e Scikit-learn. A metodologia seguida baseou-se no processo Knowledge Discovery in Databases (KDD), composto por etapas sequenciais voltadas à descoberta de padrões úteis em grandes volumes de dados (3,4).
	Na etapa de seleção, foram utilizados dados do Sinan referentes aos anos de 2023, 2024 e ao primeiro trimestre de 2025. Considerou-se a data de início dos sintomas como referência temporal para a agregação semanal dos casos, com a inclusão exclusiva de pacientes residentes no município de Osasco.
	O pré-processamento envolveu a limpeza dos dados brutos, com a exclusão de registros com classificação final “descartado” ou “em aberto”. Um dos principais desafios encontrados foi a ausência de padronização no preenchimento do campo "bairro", frequentemente sujeito a erros de digitação e abreviações variadas. Para contornar esse problema, foi construída uma lista de bairros oficiais do município e aplicada a técnica de fuzzy matching com a biblioteca rapidfuzz, estabelecendo um limite de similaridade de 85% para a correspondência.
	A fase de transformação dos dados compreendeu a redução da dimensionalidade do conjunto de dados, mantendo-se apenas os campos relevantes: bairro de residência, data do início dos sintomas e data da notificação. Dados pessoais foram completamente excluídos nesta etapa, em conformidade com os princípios éticos e legais de privacidade.
	Para a mineração de dados, foi utilizado o modelo Random Forest Regressor, escolhido por sua robustez e capacidade de lidar com dados não lineares. A base foi dividida em 70% para treinamento e 30% para teste, considerando os dados dos anos de 2023 e 2024. Com o modelo treinado, foi então realizada a previsão do número de casos para o ano de 2025, distribuídos por bairro e por semana epidemiológica.
	A avaliação do desempenho do modelo foi feita por meio do Erro Absoluto Médio (MAE) e do coeficiente de determinação (R²). Os resultados preliminares obtidos apontaram um R² de aproximadamente 0,678 e um MAE de cerca de 13,12 casos por bairro/semana. Esses valores indicam um bom ajuste do modelo, embora haja margem para aprimoramentos, especialmente no que se refere à inserção de variáveis ambientais e entomológicas.





5. REFERÊNCIAS 


Centro de Vigilância Epidemiológica. Diretrizes para prevenção e controle das arboviroses urbanas no Estado de São Paulo [Internet]. São Paulo: Secretaria de Estado da Saúde de SP; [data de publicação/atualização ou [s.d.] se não houver]. Acesso em: 28 maio 2025. Disponível em: https://portal.saude.sp.gov.br/resources/cve-centro-de-vigilancia-epidemiologica/areas-de-vigilancia/doencas-de-transmissao-por-vetores-e-zoonoses/doc/arboviroses/diretrizesparaaprevencaaoecontroledasarbovirosesurban.pdf
Brasil. Ministério da Saúde. Secretaria de Vigilância em Saúde e Ambiente. Departamento de Ações Estratégicas de Epidemiologia e Vigilância em Saúde e Ambiente. Guia de vigilância em saúde: volume 2. 6. Brasília: Ministério da Saúde; 2024. Disponível em: https://bvsms.saude.gov.br/bvs/publicacoes/guia_vigilancia_saude_v2_6edrev.pdf
Fayyad UM, Piatetsky-Shapiro G, Smyth P. From Data Mining to Knowledge Discovery in Databases. AI Mag. 1996;17(3):37–54.
Faceli K, Lorena AC, Almeida TA, Carvalho ACPLF. Inteligência Artificial – Uma Abordagem de Aprendizado de Máquina. 2. São Paulo: Grupo GEN; 2021.
