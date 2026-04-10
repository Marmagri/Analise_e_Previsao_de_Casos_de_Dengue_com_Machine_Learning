import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


def carregar_e_limpar(caminho_arquivo):
    df = pd.read_excel(caminho_arquivo)
    df.columns = df.columns.str.upper().str.strip()
    df['DT_SIN_PRI'] = pd.to_datetime(df['DT_SIN_PRI'], dayfirst=True, errors='coerce')
    df['NM_BAIRRO'] = df['NM_BAIRRO'].astype(str).str.upper().str.strip()
    df = df.dropna(subset=['DT_SIN_PRI', 'NM_BAIRRO'])
    return df


# Padronização dos bairros de Osasco

bairros_oficiais = [
    'ADALGISA', 'AYROSA', 'BARONESA', 'BELA VISTA', 'BONFIM', 'BUSSOCABA', 'CENTRO', 'CIDADE DAS FLORES',
    'CIPAVA', 'CITY BUSSOCABA', 'CONCEIÇÃO', 'CONTINENTAL', 'HELENA MARIA', 'INDUSTRIAL ANHANGUERA',
    'INDUSTRIAL MAZZEI', 'INDUSTRIAL AUTONOMISTAS', 'INDUSTRIAL REMÉDIOS', 'INDUSTRIAL ALTINO', 'IAPI',
    'JAGUARIBE', 'JARDIM DAS FLORES', "JARDIM D'ABRIL", 'JARDIM ELVIRA', 'JARDIM ROBERTO', 'KM 18',
    'METALÚRGICOS', 'MUNHOZ JUNIOR', 'MUTINGA', 'NOVO OSASCO', 'PADROEIRA', 'PESTANA', 'PIRATININGA',
    'PRESIDENTE ALTINO', 'QUITAÚNA', 'REMEDIOS', 'ROCHDALE', 'SANTA MARIA', 'SANTO ANTÔNIO', 'SÃO PEDRO',
    'UMUARAMA', 'VELOSO', 'VILA CAMPESINA', 'VILA MENCK', 'VILA OSASCO', 'VILA YARA', 'VILA YOLANDA'
]


def corrigir_bairros(df, coluna='NM_BAIRRO', lista_oficial=bairros_oficiais, limite_similaridade=85):
    bairros_corrigidos = {}
    for bairro in df[coluna].unique():
        match = process.extractOne(bairro, lista_oficial, scorer=fuzz.token_sort_ratio)
        if match:
            nome_corrigido, score, _ = match
            bairros_corrigidos[bairro] = nome_corrigido if score >= limite_similaridade else bairro
        else:
            bairros_corrigidos[bairro] = bairro
    df[coluna] = df[coluna].map(bairros_corrigidos)
    return df


# 1) Carregamento das bases de dados derivadas do Banco do SINAN

df_2023 = carregar_e_limpar("DENGUE2023.xlsx")
df_2024 = carregar_e_limpar("DENGUE2024.xlsx")
df_2025 = carregar_e_limpar("DENGUE20252.xlsx")  # novo arquivo

df = pd.concat([df_2023, df_2024, df_2025], ignore_index=True)
df = corrigir_bairros(df)


# 2) Agrupar por bairro, ano e mês

df['ano'] = df['DT_SIN_PRI'].dt.year
df['mes'] = df['DT_SIN_PRI'].dt.month

dados_agrupados = df.groupby(['NM_BAIRRO', 'ano', 'mes']).size().reset_index(name='casos')


# 3) Criação da variável de sazonalidade

dados_agrupados['mes_sin'] = np.sin(2 * np.pi * dados_agrupados['mes'] / 12)
dados_agrupados['mes_cos'] = np.cos(2 * np.pi * dados_agrupados['mes'] / 12)
dados_agrupados['temporada_risco'] = dados_agrupados['mes'].isin([10,11,12,1,2,3,4,5]).astype(int)


# 4) Separar treino e teste

df_treino = dados_agrupados[dados_agrupados['ano'].isin([2023, 2024])]
df_2025_real = dados_agrupados[dados_agrupados['ano'] == 2025]

# One-hot encoding dos bairros
df_treino_encoded = pd.get_dummies(df_treino, columns=['NM_BAIRRO'])
df_2025_encoded = pd.get_dummies(df_2025_real, columns=['NM_BAIRRO'])

# Alinhar colunas
df_2025_encoded, df_treino_encoded = df_2025_encoded.align(df_treino_encoded, join='left', axis=1, fill_value=0)


# 5) Treinar modelo

X = df_treino_encoded.drop(columns=['casos'])
y = df_treino_encoded['casos']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Avaliação por bairro
y_pred = modelo.predict(X_test)
print("MAE por bairro:", mean_absolute_error(y_test, y_pred))
print("R² por bairro:", r2_score(y_test, y_pred))


# 6) Previsão para 2025

X_2025 = df_2025_encoded.drop(columns=['casos'])
df_2025_encoded['casos_previstos'] = modelo.predict(X_2025)
df_2025_encoded[['NM_BAIRRO', 'mes']] = df_2025_real[['NM_BAIRRO', 'mes']]


# 7) Agregar por mês para métricas municipais

mensal_2025 = df_2025_encoded.groupby('mes')[['casos','casos_previstos']].sum().reset_index()

# MAE e R² municipal
mae_municipal = mean_absolute_error(mensal_2025['casos'], mensal_2025['casos_previstos'])
r2_municipal = r2_score(mensal_2025['casos'], mensal_2025['casos_previstos'])
print("MAE municipal:", mae_municipal)
print("R² municipal:", r2_municipal)


# 8) Filtrar até julho e formatar meses
-
mensal_2025 = mensal_2025[mensal_2025['mes'] <= 7]
meses_pt = {1: "Janeiro", 2: "Fevereiro", 3: "Março", 4: "Abril", 5: "Maio", 6: "Junho", 7: "Julho"}
mensal_2025['mes_nome'] = mensal_2025['mes'].map(meses_pt)


# 9) Gráfico de linha com resultados para comparativo

plt.figure(figsize=(10, 6))
sns.lineplot(data=mensal_2025, x='mes_nome', y='casos',
             label='Casos Reais', marker='o', color='blue', linewidth=3)
sns.lineplot(data=mensal_2025, x='mes_nome', y='casos_previstos',
             label='Casos Previstos', marker='o', color='red', linewidth=3)

plt.title('Casos de Dengue em 2025 (Jan-Jul): Reais vs Previstos', fontsize=14)
plt.xlabel('Mês')
plt.ylabel('Número de Casos')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
