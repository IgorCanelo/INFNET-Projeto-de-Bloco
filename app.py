import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import time
import os
import requests
import shutil
from bs4 import BeautifulSoup
from zipfile import ZipFile
from fastapi import FastAPI, HTTPException
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import yfinance as yf
import openai
from openai import OpenAI
from typing import List, Union, TypedDict
from pydantic import BaseModel, ValidationError
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv('env/config.txt')
api_key = os.getenv('API_KEY')


def inicial():

    st.title("Recomendação de FII's")
    st.markdown("""
## Objetivos do Projeto

- **Desenvolver** um modelo de recomendação de Fundos de Investimento Imobiliário (FII's).
- **Auxiliar** investidores com pouca ou nenhuma experiência.
- **Melhorar** os ganhos financeiros e otimizar o tempo dos usuários.
""")
    st.markdown("""
### Inspiração

Para mais informações e inspiração, visite o [Ranking de FIIs do Funds Explorer](https://www.fundsexplorer.com.br/ranking).
""")


def data():
    st.subheader("Amostra dos dados que serão utilizados no modelo:")
    st.write("Dataset de 2024")
    df_2024 = pd.read_csv("data/inf_mensal_fii_2024/inf_mensal_fii_complemento_2024.csv", delimiter=";", encoding="ISO-8859-1")
    st.dataframe(df_2024)



def analise_exploratoria_v1():
    st.write("Os dados obtidos são da CVM e estão divididos por ano e em cada ano possuímos três datasets, onde são informações pertinentes a ativo vs passivo, complemento e informações gerais dos fundos.")
    st.markdown("""
## Passos da análise exploratória

- **Concatenação** dos datasets de todos os anos, resultando em apenas 3 datasets finais.
- **Análise de ativos e passivos** Uma visualização para identificar quais fundos possuem mais direitos do que obrigações.
- **Segmento dos FII's** que mais possuem atuação no Brasil.
- **Dividend Yield** comparação anual.
""")
    
@st.cache_data
def concatenacao_at_pas():
    df_2020_at_pas = pd.read_csv("data/inf_mensal_fii_2020/inf_mensal_fii_ativo_passivo_2020.csv", delimiter=";", encoding="ISO-8859-1")
    df_2021_at_pas = pd.read_csv("data/inf_mensal_fii_2021/inf_mensal_fii_ativo_passivo_2021.csv", delimiter=";", encoding="ISO-8859-1")
    df_2022_at_pas = pd.read_csv("data/inf_mensal_fii_2022/inf_mensal_fii_ativo_passivo_2022.csv", delimiter=";", encoding="ISO-8859-1")
    df_2023_at_pas = pd.read_csv("data/inf_mensal_fii_2023/inf_mensal_fii_ativo_passivo_2023.csv", delimiter=";", encoding="ISO-8859-1")
    df_2024_at_pas = pd.read_csv("data/inf_mensal_fii_2024/inf_mensal_fii_ativo_passivo_2024.csv", delimiter=";", encoding="ISO-8859-1")
    df_at_pas_concat = pd.concat([df_2020_at_pas, df_2021_at_pas, df_2022_at_pas, df_2023_at_pas, df_2024_at_pas]) 

    return df_at_pas_concat


@st.cache_data
def concatenacao_complement():
    df_2020_complement = pd.read_csv("data/inf_mensal_fii_2020/inf_mensal_fii_complemento_2020.csv", delimiter=";", encoding="ISO-8859-1")
    df_2021_complement = pd.read_csv("data/inf_mensal_fii_2021/inf_mensal_fii_complemento_2021.csv", delimiter=";", encoding="ISO-8859-1")
    df_2022_complement = pd.read_csv("data/inf_mensal_fii_2022/inf_mensal_fii_complemento_2022.csv", delimiter=";", encoding="ISO-8859-1")
    df_2023_complement = pd.read_csv("data/inf_mensal_fii_2023/inf_mensal_fii_complemento_2023.csv", delimiter=";", encoding="ISO-8859-1")
    df_2024_complement = pd.read_csv("data/inf_mensal_fii_2024/inf_mensal_fii_complemento_2024.csv", delimiter=";", encoding="ISO-8859-1")
    df_complement_concat = pd.concat([df_2020_complement, df_2021_complement, df_2022_complement, df_2023_complement, df_2024_complement])

    return df_complement_concat


@st.cache_data
def concatenacao_geral():
    df_2020_geral = pd.read_csv("data/inf_mensal_fii_2020/inf_mensal_fii_geral_2020.csv", delimiter=";", encoding="ISO-8859-1")
    df_2021_geral = pd.read_csv("data/inf_mensal_fii_2021/inf_mensal_fii_geral_2021.csv", delimiter=";", encoding="ISO-8859-1")
    df_2022_geral = pd.read_csv("data/inf_mensal_fii_2022/inf_mensal_fii_geral_2022.csv", delimiter=";", encoding="ISO-8859-1")
    df_2023_geral = pd.read_csv("data/inf_mensal_fii_2023/inf_mensal_fii_geral_2023.csv", delimiter=";", encoding="ISO-8859-1")
    df_2024_geral = pd.read_csv("data/inf_mensal_fii_2024/inf_mensal_fii_geral_2024.csv", delimiter=";", encoding="ISO-8859-1")
    df_geral_concat = pd.concat([df_2020_geral, df_2021_geral, df_2022_geral, df_2023_geral, df_2024_geral])

    return df_geral_concat




def metrica_at_pas_v1(funcao):
    st.title("Análise de ativos e passivos")


    # Barra de progresso
    progress_container = st.empty()
    progress_bar = progress_container.progress(0)
    progress_container.write("Carregando dados...")
    time.sleep(3)


    # Trazendo o df concatenado
    df_v1 = funcao()
    progress_bar.progress(20)
    

    # Cálculo de ativo com base no balanço contábil
    df_v1['Total_Ativo'] = df_v1[['Total_Necessidades_Liquidez',
                                  'Total_Investido',
                                  'Direitos_Bens_Imoveis',
                                  'Valores_Receber'
                                  ]].sum(axis=1)
    progress_bar.progress(40)


    # Transformação para datetime, e obtendo apenas o ano
    df_v1["Data_Referencia"] = pd.to_datetime(df_v1["Data_Referencia"], format="%Y-%m-%d")
    df_v1["Ano"] = df_v1["Data_Referencia"].dt.year
    progress_bar.progress(60)


    # Agregando os valores por fundo e ano e realizando a soma de passivos e ativos e utilizando apenas os top 5 para melhor visualização
    ativos_por_fundo = df_v1.groupby(["CNPJ_Fundo", "Ano"])["Total_Ativo"].sum().reset_index()
    passivos_por_fundo = df_v1.groupby(["CNPJ_Fundo", "Ano"])["Total_Passivo"].sum().reset_index()
    maiores_ativos = ativos_por_fundo.groupby("Ano").apply(lambda x: x.nlargest(5, "Total_Ativo")).reset_index(drop=True)
    maiores_passivos = passivos_por_fundo.groupby("Ano").apply(lambda x: x.nlargest(5, "Total_Passivo")).reset_index(drop=True)
    progress_bar.progress(80)


    # Gráfico ativos
    st.subheader("Gráfico de ativos")
    st.write("Top 5 maiores ativos por ano")
    fig_ativos = px.bar(maiores_ativos, 
                    x='Ano', 
                    y='Total_Ativo', 
                    color='CNPJ_Fundo', 
                    barmode='group',
                    title='Total de Ativos dos Fundos por Ano',
                    labels={'Total_Ativo': 'Total de Ativos', 'Ano': 'Ano'},
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    height=500)

    fig_ativos.update_layout(title={'x': 0.5},
                            xaxis_title='Ano',
                            yaxis_title='Total de Ativos',
                            template='plotly_white',
                            legend_title_text='Fundo')
    
    st.plotly_chart(fig_ativos)


    # Gráfico passivos
    st.subheader("Gráfico de passivos")
    st.write("Top 5 maiores passivos por ano")
    fig_passivos = px.bar(maiores_passivos, 
                      x='Ano', 
                      y='Total_Passivo', 
                      color='CNPJ_Fundo', 
                      barmode='group',
                      title='Total de Passivos por Fundo por Ano',
                      labels={'Total_Passivo': 'Total de Passivos', 'Ano': 'Ano'},
                      color_discrete_sequence=px.colors.qualitative.Set2,
                      height=500)
    
    fig_passivos.update_layout(title={'x': 0.5},
                            xaxis_title='Ano',
                            yaxis_title='Total de Passivos',
                            template='plotly_white',
                            legend_title_text='Fundo')
    
    st.plotly_chart(fig_passivos)
    

    # Finalização da barra de progresso e exclusão da mesma
    progress_bar.progress(100)
    progress_container.empty()




def segmento_fiis(funcao):
    st.title("Quantidade de FII's em cada segmento por ano")

    # Trazendo o df concatenado e ano
    df_v2 = funcao()
    df_v2["Data_Referencia"] = pd.to_datetime(df_v2["Data_Referencia"], format="%Y-%m-%d")
    df_v2["Ano"] = df_v2["Data_Referencia"].dt.year

    # Agrupar por Segmento e Ano
    segmentos = df_v2.groupby(["Segmento_Atuacao", "Ano"])["Segmento_Atuacao"].count().reset_index(name="Quantidade_Segmentos")
    

    anos = segmentos['Ano'].unique()
    for ano in anos:
        df_ano = segmentos[segmentos['Ano'] == ano]

        # Gráfico de barras usando Plotly
        fig = px.bar(df_ano, 
                     x='Segmento_Atuacao', 
                     y='Quantidade_Segmentos', 
                     color='Segmento_Atuacao', 
                     title=f'Quantidade de Segmentos de Atuação - {ano}',
                     labels={'Quantidade_Segmentos': 'Quantidade de Segmentos', 'Segmento_Atuacao': 'Segmento de Atuação'},
                     color_discrete_sequence=px.colors.qualitative.Set2,
                     height=500)

        fig.update_layout(title={'x':0.5},
                          xaxis_title='Segmento de Atuação', 
                          yaxis_title='Quantidade de Segmentos',
                          template='plotly_white')

        st.plotly_chart(fig)



def scatter_plot(funcao):
    st.title("Dividend Yield ao longo dos anos, observando os top 5 fundos com mais DY")

    # Ajustes nos dados
    df_v3 = funcao()
    df_v3["Data_Referencia"] = pd.to_datetime(df_v3["Data_Referencia"], format="mixed")
    df_v3["Ano"] = df_v3["Data_Referencia"].dt.year
    df_v3["Percentual_Dividend_Yield_Mes"] = pd.to_numeric(df_v3["Percentual_Dividend_Yield_Mes"], errors='coerce')
    dividend_yield = df_v3.groupby(["CNPJ_Fundo", "Ano"])["Percentual_Dividend_Yield_Mes"].sum().reset_index(name="Percentual_Dividend_Yield_Ano")
    top_5_fundos = dividend_yield.groupby("CNPJ_Fundo")["Percentual_Dividend_Yield_Ano"].sum().nlargest(5).index
    
    # Cálculo das métricas para os top 5 fundos
    top_5_metrics = dividend_yield[dividend_yield['CNPJ_Fundo'].isin(top_5_fundos)].groupby("CNPJ_Fundo")["Percentual_Dividend_Yield_Ano"].agg(['sum', 'mean']).reset_index()
    top_5_metrics.columns = ['CNPJ_Fundo', 'Soma_Dividend_Yield', 'Media_Dividend_Yield']


    # Métricas
    st.subheader("Métricas dos Top 5 Fundos")
    for index, row in top_5_metrics.iterrows():
        st.metric(label=row['CNPJ_Fundo'], value=f"DY anual: {row['Soma_Dividend_Yield']:.2f}", delta=f"Média: {row['Media_Dividend_Yield']:.2f}")


    # Gráfico scatter plot
    dividend_yield['Cor'] = dividend_yield['CNPJ_Fundo'].apply(lambda x: 'Top 5' if x in top_5_fundos else 'Outros')
    fig = px.scatter(
        dividend_yield,
        x='Ano',
        y='Percentual_Dividend_Yield_Ano',
        color='Cor',
        hover_name='CNPJ_Fundo',
        hover_data=['Percentual_Dividend_Yield_Ano'],
        color_discrete_map={'Top 5': 'red', 'Outros': 'blue'},
        labels={'Percentual_Dividend_Yield_Ano': 'Dividend Yield'},
        title='Dividend Yield por Fundo Imobiliário e Ano'
    )

    fig.update_layout(
        xaxis_title='Ano',
        yaxis_title='Dividend Yield',
        legend_title='Legenda',
        height=600,
        width=800
    )

    fig.update_yaxes(range=[0, 30])
    st.plotly_chart(fig)




def datasets_download(funcao_1, funcao_2, funcao_3):
    st.subheader("Os datasets finais foram três, quais deseja realizar o download?")

    df_v5 = funcao_1()
    df_v6 = funcao_2()
    df_v7 = funcao_3()

    options = {
        "Dataset Ativos e Passivos": df_v5,
        "Dataset Complemento": df_v6,
        "Dataset Geral": df_v7,
    }

    # Checkboxes para escolher os datasets
    selected_options = []
    for option in options.keys():
        if st.checkbox(f"Selecionar {option}"):
            selected_options.append(option)


    # Exibir os datasets selecionados e permitir a filtragem das colunas
    filtered_dataframes = {}
    for option in selected_options:
        st.subheader(option)
        df = options[option]
        
        # Pergunta se o usuário quer filtrar o dataset
        if st.checkbox(f"Filtrar {option}?"):
            data_referencia = st.date_input("Selecione uma Data de Referência:", value=None, key=f"date_input_{option}")
            cnpj_fundo = st.text_input("Digite o CNPJ do Fundo:", value="", key=f"cnpj_input_{option}")
            
            if data_referencia:
                df = df[df['Data_Referencia'] == data_referencia]
            if cnpj_fundo:
                df = df[df['CNPJ_Fundo'] == cnpj_fundo]

        # Permitir que o usuário escolha quais colunas exibir
        columns_to_select = st.multiselect(
            f"Selecione as colunas para {option}:",
            df.columns.tolist(),
            default=df.columns.tolist()
        )

        filtered_df = df[columns_to_select]
        st.dataframe(filtered_df)

        # Armazena o DataFrame filtrado para download
        filtered_dataframes[option] = filtered_df

    # Botão para download
    if st.button("Baixar Datasets Selecionados"):
        with st.spinner("Baixando dados..."):
            time.sleep(3)
            for option in filtered_dataframes.keys():
                csv = filtered_dataframes[option].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"Baixar {option} como CSV",
                    data=csv,
                    file_name=f'{option}.csv',
                    mime='text/csv',
                    key=f"download_{option}"
                )




def atualizar_dados_scrapping():

    def baixar_arquivo_zip():

        url = "https://dados.cvm.gov.br/dataset/fii-doc-inf_mensal"
        response = requests.get(url)

        links = [] 
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            for link in soup.find_all('a', href=True):
                href = link['href']
                if 'inf_mensal_fii_2024.zip' in href:
                    print(f"Link encontrado: {href}")
                    links.append(href)
        else:
            print(f"Falha ao acessar a página. Código de status: {response.status_code}")
            return None

        if links:
            output_file = "data/inf_mensal_fii_2024.zip"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            download_url = links[0]
            response = requests.get(download_url)

            if response.status_code == 200:
                with open(output_file, 'wb') as file:
                    file.write(response.content)
                print(f"Download concluído e salvo como {output_file}")
                return output_file
            else:
                print(f"Falha no download. Código de status: {response.status_code}")
                return None
        else:
            print("Nenhum link correspondente foi encontrado.")
            return None


    def limpar_pasta_destino(pasta_destino):
        if os.path.exists(pasta_destino):
            shutil.rmtree(pasta_destino)  
        os.makedirs(pasta_destino)

    def extrair_arquivo_zip(arquivo_zip, pasta_destino):
        with ZipFile(arquivo_zip, 'r') as zip_ref:
            zip_ref.extractall(pasta_destino)
        print(f"Arquivos extraídos para {pasta_destino}")

    def excluir_arquivo_zip(arquivo_zip):
        if os.path.exists(arquivo_zip):
            os.remove(arquivo_zip)
            print(f"Arquivo ZIP {arquivo_zip} excluído.")


    pasta_destino = "data/inf_mensal_fii_2024"


    def atualizar_dados():

        arquivo_zip = baixar_arquivo_zip()
        if arquivo_zip:
            limpar_pasta_destino(pasta_destino)
            extrair_arquivo_zip(arquivo_zip, pasta_destino)
            excluir_arquivo_zip(arquivo_zip)

    
    st.subheader("Clique no botão abaixo para atualizar todo o conjunto de dados do modelo")
    if st.button("Atualizar Conjunto de Dados"):
        atualizar_dados()
        st.write("Conjunto de Dados Atualizado:")
    

def recomendacao_inicial():
    st.markdown("""
## Quais são os principais perfis de investidores?

- **Conservador** - Evita riscos e prefere investimentos com menor retorno e maior segurança.
- **Moderado** - Aceita mais riscos do que o conservador, mas volta a investir em opções mais seguras em momentos de instabilidade.
- **Arrojado** - Tem maior apetite ao risco e busca maiores retornos financeiros.
""")
    opcoes = ['Conservador', 'Moderado', 'Arrojado']
    perfil_investidor = st.selectbox("Selecione o seu perfil de investidor", opcoes)
    st.write(f'Você selecionou: {perfil_investidor}')
    st.session_state["investidor"] = perfil_investidor
    return perfil_investidor

def endpoints_api():
    st.subheader("Endpoints utilizados para obter informações via API")
    st.write("Para realizar testes e se familiarizar com os endpoints da API recomendamos acessar o Swagger no endpoint abaixo:")
    st.write("http://127.0.0.1:8000/docs#/")
    st.subheader("POST - Conversar com especialista virtual focado em FII's")
    st.write("Esse endpoint é utilizado para obter uma conversação com assistente financeiro virtual para tirar dúvidas pertinentes a FII's")
    st.write("http://127.0.0.1:8000/chat/especialista_fii")
    st.subheader("GET - Obter as informações de ativos e passivos por CNPJ")
    st.write("Esse endpoint trás todas as informações de ativos e passivos do conjunto de dados desde 2020 ate a data mais recente, via CNPJ como parâmetro da requisição com apenas números.")
    st.write("http://127.0.0.1:8000/dataset/ativos_passivos/{cnpj}")
    st.subheader("GET - Obter as informações de complemento por CNPJ")
    st.write("Esse endpoint trás todas as informações de complemento do conjunto de dados desde 2020 ate a data mais recente, via CNPJ como parâmetro da requisição com apenas números.")
    st.write("http://127.0.0.1:8000/dataset/complemento/{cnpj}")
    st.subheader("GET - Obter as informações gerais por CNPJ")
    st.write("Esse endpoint trás todas as informações gerais do conjunto de dados desde 2020 ate a data mais recente, via CNPJ como parâmetro da requisição com apenas números.")
    st.write("http://127.0.0.1:8000/dataset/geral/{cnpj}")


def recomendacao_inicial_investidor():
    st.markdown("""
## Como você se classifica em relação a investimento em Fundos Imobiliários (FIIs)?

- **Iniciante** - Tenho pouco ou nenhum conhecimento sobre investimentos em FIIs.
- **Intermediário** - Já investi em FIIs, mas ainda tenho poucos conhecimentos sobre o tema.
- **Avançado** - Conheço o mercado de FIIs, possuo capital alocado e acompanho o desempenho regularmente.
""")
    opcoes = ['Iniciante', 'Intermediário', 'Avançado']
    investidor = st.selectbox("Selecione o seu nível de conhecimento em FIIs", opcoes)
    st.write(f'Você selecionou: {investidor}')
    st.session_state["segmentos"] = investidor
    return investidor

def valor_disposto_investir():

    st.markdown("""
## Qual o valor que você tem em mente para investir por cota?
""")

    valor_investido = st.selectbox(
        "Qual o valor que você tem em mente para investir por cota?",
        ["R$0,00 a R$90,00", "R$91,00 a R$120,00", "Acima de R$121,00", "Não possuo valor pré-estabelecido"]
    )
    st.session_state["segmentos"] = valor_investido
    return valor_investido

def recomendacao_historica():

    st.markdown("""
## Você deseja obter essa recomendação com base em:

- **Histórica** - Desde a criação do FII.  
- **Anual** - O ano atual.
- **Mensal** - O mês atual.    
""")
    opcoes = ['Histórica', 'Anual', 'Mensal']
    historico = st.radio("Selecione o período desejado para obter a recomendação", opcoes)
    st.write(f'Você selecionou: {historico}')
    st.session_state["segmentos"] = historico 
    return historico


def recomendacao_segmento():
    st.markdown("""
## Selecione os segmentos dos FIIs a serem recomendados:
""")
    df = concatenacao_geral()
    df["Segmento_Atuacao"] = df["Segmento_Atuacao"].fillna("Outros")
    lista_segmentos = df["Segmento_Atuacao"].unique().tolist()

    segmentos = st.multiselect(
    "Selecione as opções desejadas:",
    lista_segmentos
)
    st.session_state["segmentos"] = segmentos
    st.write("Segmentos selecionados:", segmentos)
    return segmentos



def recomendacao_quantidade():
    st.markdown("""
## Selecione quantos FIIs deseja obter de recomendação:
""")
    lista_qtde = [1, 2, 3, 4, 5]

    quantidade_fii = st.selectbox(
    "Selecione as opções desejadas:",
    lista_qtde
)
    st.session_state["quantidade"] = quantidade_fii
    return quantidade_fii



def score_df():
    # Obtendo os DFs
    metrica_1 = concatenacao_complement()
    df_segmento = concatenacao_geral()

    # Trazendo os tickers de todos os FIIs listados na bolsa atualmente
    df_fiis_listados_atuais = pd.read_csv(r"data\Tickers\cnpj_fundos.csv", sep=";")
    df_fiis_listados_atuais = df_fiis_listados_atuais.rename(columns={'CNPJ': 'CNPJ_Fundo'})

    # Obtendo apenas os FIIs listados atualmente
    metrica_1 = metrica_1.merge(df_fiis_listados_atuais[["TICKER", "CNPJ_Fundo"]], how='left')
    metrica_1 = metrica_1[~metrica_1['TICKER'].isna()]

    # Seleção das colunas necessárias
    metrica_1 = metrica_1[["CNPJ_Fundo", "Data_Referencia", "TICKER", "Percentual_Rentabilidade_Efetiva_Mes", "Percentual_Rentabilidade_Patrimonial_Mes", "Percentual_Dividend_Yield_Mes"]]

    # Tratamento no dataset em relação as valores
    metrica_1["Percentual_Rentabilidade_Efetiva_Mes"] = metrica_1["Percentual_Rentabilidade_Efetiva_Mes"].replace({'.': ''}, regex=True)
    metrica_1["Percentual_Rentabilidade_Efetiva_Mes"] = metrica_1["Percentual_Rentabilidade_Efetiva_Mes"].replace({',': '.'}, regex=True)
    metrica_1["Percentual_Rentabilidade_Efetiva_Mes"] = metrica_1["Percentual_Rentabilidade_Efetiva_Mes"].replace('', '0')
    metrica_1["Percentual_Rentabilidade_Efetiva_Mes"] = metrica_1["Percentual_Rentabilidade_Efetiva_Mes"].fillna(0)

    metrica_1["Percentual_Rentabilidade_Patrimonial_Mes"] = metrica_1["Percentual_Rentabilidade_Patrimonial_Mes"].replace({'.': ''}, regex=True)
    metrica_1["Percentual_Rentabilidade_Patrimonial_Mes"] = metrica_1["Percentual_Rentabilidade_Patrimonial_Mes"].replace({',': '.'}, regex=True)
    metrica_1["Percentual_Rentabilidade_Patrimonial_Mes"] = metrica_1["Percentual_Rentabilidade_Patrimonial_Mes"].replace('', '0')
    metrica_1["Percentual_Rentabilidade_Patrimonial_Mes"] = metrica_1["Percentual_Rentabilidade_Patrimonial_Mes"].fillna(0)

    metrica_1["Percentual_Dividend_Yield_Mes"] = metrica_1["Percentual_Dividend_Yield_Mes"].replace({'.': ''}, regex=True)
    metrica_1["Percentual_Dividend_Yield_Mes"] = metrica_1["Percentual_Dividend_Yield_Mes"].replace({',': '.'}, regex=True)
    metrica_1["Percentual_Dividend_Yield_Mes"] = metrica_1["Percentual_Dividend_Yield_Mes"].replace('', '0')
    metrica_1["Percentual_Dividend_Yield_Mes"] = metrica_1["Percentual_Dividend_Yield_Mes"].fillna(0)

    # Trasnformando em float
    metrica_1["Percentual_Rentabilidade_Efetiva_Mes"] = metrica_1["Percentual_Rentabilidade_Efetiva_Mes"].astype(float) * 100
    metrica_1["Percentual_Rentabilidade_Patrimonial_Mes"] = metrica_1["Percentual_Rentabilidade_Patrimonial_Mes"].astype(float) * 100
    metrica_1["Percentual_Dividend_Yield_Mes"] = metrica_1["Percentual_Dividend_Yield_Mes"].astype(float) * 100

    # Filtrando valores
    metrica_1.loc[metrica_1['Percentual_Rentabilidade_Efetiva_Mes'] < 0, 'Percentual_Rentabilidade_Efetiva_Mes'] = 0
    metrica_1.loc[metrica_1['Percentual_Rentabilidade_Patrimonial_Mes'] < 0, 'Percentual_Rentabilidade_Patrimonial_Mes'] = 0
    metrica_1.loc[metrica_1['Percentual_Dividend_Yield_Mes'] < 0, 'Percentual_Dividend_Yield_Mes'] = 0

    # Normalização dos dados
    scaler = MinMaxScaler(feature_range=(0, 100))
    metrica_1[['Percentual_Rentabilidade_Efetiva_Mes', 'Percentual_Rentabilidade_Patrimonial_Mes', 'Percentual_Dividend_Yield_Mes']] = scaler.fit_transform(
        metrica_1[['Percentual_Rentabilidade_Efetiva_Mes', 'Percentual_Rentabilidade_Patrimonial_Mes', 'Percentual_Dividend_Yield_Mes']]
    )

    # Cálculo de score para cada tipo de perfil de investidor
    metrica_1['score1_conservador'] = (0.6 * metrica_1["Percentual_Dividend_Yield_Mes"]) + (0.3 * metrica_1["Percentual_Rentabilidade_Efetiva_Mes"]) + (0.1 * metrica_1["Percentual_Rentabilidade_Patrimonial_Mes"])
    metrica_1['score1_conservador'] = metrica_1['score1_conservador'].apply(lambda x: f"{x:.2f}")
    metrica_1['score1_moderado'] = (0.4 * metrica_1["Percentual_Dividend_Yield_Mes"]) + (0.4 * metrica_1["Percentual_Rentabilidade_Efetiva_Mes"]) + (0.2 * metrica_1["Percentual_Rentabilidade_Patrimonial_Mes"])
    metrica_1['score1_moderado'] = metrica_1['score1_moderado'].apply(lambda x: f"{x:.2f}")
    metrica_1['score1_arrojado'] = (0.2 * metrica_1["Percentual_Dividend_Yield_Mes"]) + (0.5 * metrica_1["Percentual_Rentabilidade_Efetiva_Mes"]) + (0.3 * metrica_1["Percentual_Rentabilidade_Patrimonial_Mes"])
    metrica_1['score1_arrojado'] = metrica_1['score1_arrojado'].apply(lambda x: f"{x:.2f}")

    # Deixando apenas as colunas calculadas para diminuir o tamanho do dataset
    metrica_1 = metrica_1[["CNPJ_Fundo", "Data_Referencia", "TICKER", "score1_conservador", "score1_moderado", "score1_arrojado"]]

    # Organização da data
    metrica_1['Data_Referencia'] = pd.to_datetime(metrica_1['Data_Referencia'], dayfirst=True, format='mixed')
    
    # Trazendo o segmento dos fundos
    df_segmento = df_segmento[["CNPJ_Fundo", "Segmento_Atuacao"]]
    df_segmento = df_segmento.drop_duplicates(subset=["CNPJ_Fundo"], keep="first")
    metrica_1 = metrica_1.merge(df_segmento[["CNPJ_Fundo", "Segmento_Atuacao"]], how="left", on="CNPJ_Fundo")
    metrica_1["Segmento_Atuacao"] = metrica_1["Segmento_Atuacao"].fillna("Outros")
 
    # Filtrando o DF com base nas escolhas do usuário
    perfil_investidor = st.session_state.get('tipo_investidor')
    if perfil_investidor == "Conservador":
        metrica_1 = metrica_1[["CNPJ_Fundo", "Data_Referencia", "TICKER", "Segmento_Atuacao", "score1_conservador"]]
        metrica_1 = metrica_1.sort_values(by="score1_conservador", ascending=False)

    elif perfil_investidor == "Moderado":
        metrica_1 = metrica_1[["CNPJ_Fundo", "Data_Referencia", "TICKER", "Segmento_Atuacao", "score1_moderado"]]
        metrica_1 = metrica_1.sort_values(by="score1_moderado", ascending=False)

    elif perfil_investidor == "Arrojado":
        metrica_1 = metrica_1[["CNPJ_Fundo", "Data_Referencia", "TICKER", "Segmento_Atuacao", "score1_arrojado"]]
        metrica_1 = metrica_1.sort_values(by="score1_arrojado", ascending=False)
    
    

    # Filtrando o DF com base no histórico escolhido
    historico = st.session_state.get('historico')
    ano_mais_recente = metrica_1['Data_Referencia'].dt.year.max()
    mes_mais_recente_v1 = metrica_1[metrica_1['Data_Referencia'].dt.year == ano_mais_recente]
    mes_mais_recente_v2 = mes_mais_recente_v1['Data_Referencia'].dt.month.max()
    if historico == "Anual":
        metrica_1 = metrica_1[metrica_1['Data_Referencia'].dt.year == ano_mais_recente]

    elif historico == "Mensal":
        metrica_1 = metrica_1[metrica_1['Data_Referencia'].dt.year == ano_mais_recente]
        metrica_1 = metrica_1[metrica_1['Data_Referencia'].dt.month == mes_mais_recente_v2]

    # Filtrando o DF com base nos segmentos escolhidos
    segmentos = st.session_state.get('segmentos')
    if segmentos:
        metrica_1 = metrica_1[metrica_1["Segmento_Atuacao"].isin(segmentos)]

    # Diminuindo tamanho do dataset para pegar as cotações
    metrica_1 = metrica_1.head(300)
    

    # Obter cotações de forma otimizada
    metrica_1['TICKERS_SA'] = metrica_1['TICKER'] + ".SA"
    tickers_unicos = " ".join(metrica_1['TICKERS_SA'].unique())
    
    # Obtém todas as cotações de uma vez
    cotacoes = yf.download(tickers_unicos, period="1d")['Close']
    
    # Mapeia as cotações para o DataFrame
    if isinstance(cotacoes, pd.Series):  # caso de um único ticker
        metrica_1['cotacao'] = cotacoes.iloc[-1]
    else:  # caso de múltiplos tickers
        cotacoes_dict = cotacoes.iloc[-1].to_dict()
        metrica_1['cotacao'] = metrica_1['TICKERS_SA'].map(cotacoes_dict)

    # Limpeza das cotações
    metrica_1 = metrica_1[metrica_1['cotacao'].notna()]
    metrica_1.drop(columns=['TICKERS_SA'], inplace=True)





    # Filtrando o DF com base no valor que o usuário está disposto a investir
    valor_disposto = st.session_state.get('valor_disposto')
    if valor_disposto == "R$0,00 a R$90,00":
        metrica_1 = metrica_1[metrica_1['cotacao'] <= 90.00]

    elif valor_disposto == "R$91,00 a R$120,00":
        metrica_1 = metrica_1[(metrica_1['cotacao'] > 90.00) & (metrica_1['cotacao'] <= 120.00)]

    elif valor_disposto == "Acima de R$121,00":
        metrica_1 = metrica_1[metrica_1['cotacao'] > 121.00]

    # Filtrando o DF com base na quantidade de FIIs escolhidos para recomendação
    quantidade = int(st.session_state.get('quantidade'))
    metrica_1 = metrica_1.head(quantidade)
    metrica_1 = metrica_1.drop_duplicates(subset='TICKER')

    return metrica_1

def formatar_numero(valor):
    if valor >= 1e9:  # Bilhões
        return f"{valor / 1e9:.2f} bilhões"
    elif valor >= 1e6:  # Milhões
        return f"{valor / 1e6:.2f} milhões"
    elif valor >= 1e3:  # Milhares
        return f"{valor / 1e3:.2f} mil"
    else:  # Valores menores
        return f"{valor:.2f}"
    


def gerar_analise_fii(ticker, dividend_yield, patrimonio_liquido, valor_cota, cotistas, segmento):
    openai.api_key
    
    prompt = f"""Você é um analista financeiro especializado em Fundos Imobiliários (FIIs) do Brasil.
    Analise os dados abaixo e forneça insights valiosos sobre o FII em questão.
    
    Dados do FII:
    - Ticker: {ticker}
    - Dividend Yield Mensal Atual: {dividend_yield}%
    - Patrimônio Líquido: R$ {patrimonio_liquido}
    - Valor da Cota: R$ {valor_cota}
    - Total de Cotistas: {cotistas}
    - Segmento: {segmento}
    
    Foque em:
    1. Resumo sobre o FII
    2. Avaliação do patrimônio líquido
    3. Considerações sobre o segmento
    4. Riscos e oportunidades
    
    Mantenha um tom profissional e objetivo."""

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erro ao gerar análise: {str(e)}"


def inicializar_chat():
    if "mensagens" not in st.session_state:
        st.session_state.mensagens = []
    if "openai_client" not in st.session_state:
        st.session_state.openai_client = OpenAI(api_key=api_key)

def chat_fii():
    st.subheader("Chat com especialista virtual focado em FIIs")
    st.write("Tire qualquer dúvida que tiver sobre qualquer tema voltado para o mercado brasileiro de fundos imobiliários, digite abaixo sua dúvida!")
    
    inicializar_chat()
    
    for mensagem in st.session_state.mensagens:
        with st.chat_message(mensagem["role"]):
            st.write(mensagem["content"])
    
    # Campo de input para nova mensagem
    if prompt := st.chat_input("Digite sua pergunta sobre FIIs"):
        st.session_state.mensagens.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
            
        contexto = """Você é um especialista em Fundos Imobiliários (FIIs) do mercado brasileiro.
        Forneça respostas claras e objetivas sobre FIIs, incluindo análises, recomendações e explicações
        sobre conceitos importantes do mercado. Mantenha um tom profissional e educativo."""
        
        try:
            mensagens_completas = [
                {"role": "system", "content": contexto}
            ] + st.session_state.mensagens
            
            response = st.session_state.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=mensagens_completas,
                temperature=0.7,
                max_tokens=1000
            )
            
            resposta = response.choices[0].message.content
            
            st.session_state.mensagens.append({"role": "assistant", "content": resposta})
            
            with st.chat_message("assistant"):
                st.write(resposta)
                
        except Exception as e:
            st.error(f"Erro ao gerar resposta: {str(e)}")




def escolha_analise():
    st.subheader("Com base nas suas preferências, as recomendações foram:")
    
    df = score_df()

    df_complemento = concatenacao_complement()
    df_complemento['Data_Referencia'] = pd.to_datetime(df_complemento['Data_Referencia'], dayfirst=True, format='mixed')
    
    st.write("Aqui estão os fundos imobiliários recomendados:")
    
    # CSS atualizado para exibir um card por linha
    st.markdown("""
        <style>
        .fundo-card {
            background-color: #1a1a1a;
            border: 1px solid #333;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            transition: all 0.3s ease;
            text-align: center;
            width: 100%;
        }
        .fundo-card:hover {
            border-color: #3b82f6;
            transform: translateY(-2px);
        }
        .ticker {
            font-size: 28px;
            font-weight: bold;
            color: #3b82f6;
            margin-bottom: 15px;
        }
        .setor {
            background-color: rgba(59, 130, 246, 0.1);
            color: #3b82f6;
            padding: 6px 16px;
            border-radius: 15px;
            font-size: 14px;
            display: inline-block;
            margin-bottom: 15px;
        }
        .info {
            color: #9ca3af;
            margin: 10px 0;
            font-size: 16px;
        }
        .info p {
            margin: 5px 0;
        }
        .button-container {
            margin-top: 15px;
            width: 100%;
        }
        .card {
            background-color: #1a1a1a;
            border: 1px solid #333;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            transition: all 0.3s ease;
            text-align: center;
            width: 100%;
            min-height: 150px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .card:hover {
            border-color: #3b82f6;
            transform: translateY(-2px);
        }
        .metric-value {
            font-size: 18px;  /* Reduzido de 28px */
            font-weight: bold;
            color: #3b82f6;
            margin-bottom: 15px;
            word-wrap: break-word;  /* Permite quebra de palavras */
            overflow-wrap: break-word;
            hyphens: auto;
            padding: 0 5px;
        }
        .metric-label {
            background-color: rgba(59, 130, 246, 0.1);
            color: #3b82f6;
            padding: 6px 16px;
            border-radius: 15px;
            font-size: 14px;
            display: inline-block;
            margin-bottom: 15px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Inicializar a variável na session_state se não existir
    if 'ticker_escolhido' not in st.session_state:
        st.session_state.ticker_escolhido = None
    
    # Iterar sobre o DataFrame usando índices
    for index, row in df.iterrows():
        # Container para cada card
        with st.container():
            # Card
            st.markdown(f"""
                <div class="fundo-card">
                    <div class="ticker">{row['TICKER']}</div>
                    <div class="setor">{row['Segmento_Atuacao']}</div>
                    <div class="info">
                        <p>Valor: R$ {row['cotacao']:.2f}</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Botão para cada card
            if st.button(f"Ver Detalhes de {row['TICKER']}", key=f"btn_{row['TICKER']}"):
                st.session_state.ticker_escolhido = row['TICKER']
    
    # Mostrar detalhes apenas quando um ticker for selecionado via botão
    if st.session_state.ticker_escolhido:
        with st.expander(f"Detalhes de {st.session_state.ticker_escolhido}", expanded=True):
            st.markdown(f"### Você escolheu o fundo {st.session_state.ticker_escolhido}")
            ################################################################
             
            cnpj_fundo = df[df['TICKER'] == st.session_state.ticker_escolhido]['CNPJ_Fundo'].iloc[0]
            valor_cota = df[df['TICKER'] == st.session_state.ticker_escolhido]['cotacao'].iloc[0]
            segmento_atuacao = df[df['TICKER'] == st.session_state.ticker_escolhido]['Segmento_Atuacao'].iloc[0]
            
            # Filtrando dados complementares do fundo
            dados_fundo = df_complemento[df_complemento['CNPJ_Fundo'] == cnpj_fundo]
            
            # Obtendo o mês e ano mais recentes
            ano_mais_recente = dados_fundo['Data_Referencia'].dt.year.max()
            mes_mais_recente_v1 = dados_fundo[dados_fundo['Data_Referencia'].dt.year == ano_mais_recente]
            mes_mais_recente_v2 = mes_mais_recente_v1['Data_Referencia'].dt.month.max()
            dados_recentes = mes_mais_recente_v1[mes_mais_recente_v1['Data_Referencia'].dt.month == mes_mais_recente_v2]
            
            # Obtendo os valores necessários
            dividend_yield = dados_recentes['Percentual_Dividend_Yield_Mes'].iloc[0]  # Usando iloc[0] para pegar apenas o valor
            dividend_yield = float(dividend_yield) * 100       
            
            patrimonio_liquido = dados_recentes['Patrimonio_Liquido'].iloc[0]
            cotistas = dados_recentes['Total_Numero_Cotistas'].iloc[0]
            
            # Estilização CSS para os cards
            st.markdown("""
            <style>
            .card {
                background-color: #1a1a1a;
                border: 1px solid #333;
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0;
                transition: all 0.3s ease;
                text-align: center;
                width: 100%;
            }

            .card:hover {
                border-color: #3b82f6;
                transform: translateY(-2px);
            }

            .metric-value {
                font-size: 28px;
                font-weight: bold;
                color: #3b82f6;
                margin-bottom: 15px;
            }

            .metric-label {
                background-color: rgba(59, 130, 246, 0.1);
                color: #3b82f6;
                padding: 6px 16px;
                border-radius: 15px;
                font-size: 14px;
                display: inline-block;
                margin-bottom: 15px;
            }

            .info {
                color: #9ca3af;
                margin: 10px 0;
                font-size: 16px;
            }

            .info p {
                margin: 5px 0;
            }

            .button-container {
                margin-top: 15px;
                width: 100%;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Criando o layout em colunas para os cards
            row1_col1, row1_col2 = st.columns(2)
            row2_col1, row2_col2 = st.columns(2)
            row3_col1, row3_col2 = st.columns(2)
            
            with row1_col1:
                st.markdown(f"""
                <div class="card">
                    <div class="metric-label">CNPJ do Fundo</div>
                    <div class="metric-value">{cnpj_fundo}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with row1_col2:
                st.markdown(f"""
                <div class="card">
                    <div class="metric-label">Dividend Yield</div>
                    <div class="metric-value">{round(dividend_yield, 2)}%</div>
                </div>
                """, unsafe_allow_html=True)
                
            with row2_col1:
                st.markdown(f"""
                <div class="card">
                    <div class="metric-label">Patrimônio Líquido</div>
                    <div class="metric-value">R$ {formatar_numero(patrimonio_liquido)}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with row2_col2:
                st.markdown(f"""
                <div class="card">
                    <div class="metric-label">Valor da Cota</div>
                    <div class="metric-value">R$ {valor_cota:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with row3_col1:
                st.markdown(f"""
                <div class="card">
                    <div class="metric-label">Total de Cotistas</div>
                    <div class="metric-value">{formatar_numero(cotistas)}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with row3_col2:
                st.markdown(f"""
                <div class="card">
                    <div class="metric-label">Segmento de Atuação</div>
                    <div class="metric-value">{segmento_atuacao}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Informações adicionais permanecem as mesmas
            st.markdown("### Histórico de Dividend Yield")
            mes_mais_recente_v1['Dividend_Yield_Percentual'] = round(mes_mais_recente_v1['Percentual_Dividend_Yield_Mes'].astype(float) * 100, 2)
            
            # Criar gráfico com plotly
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=mes_mais_recente_v1['Data_Referencia'],
                y=mes_mais_recente_v1['Dividend_Yield_Percentual'],
                mode='lines+markers',
                name='Dividend Yield',
                line=dict(color='#3b82f6', width=2),
                marker=dict(size=8, symbol='circle')
            ))
            
            # Customizar o layout do gráfico
            fig.update_layout(
                title={
                    'text': 'Histórico do Dividend Yield',
                    'y':0.9,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                xaxis_title='Data',
                yaxis_title='Dividend Yield (%)',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128,128,128,0.2)',
                    tickformat='%B/%Y'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128,128,128,0.2)',
                    ticksuffix='%'
                ),
                hovermode='x unified',
                margin=dict(l=60, r=30, t=80, b=60)
            )
            
            # Adicionar o gráfico ao Streamlit
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Histórico do Valor Patrimonial")
            mes_mais_recente_v1["valor_patrimonial"] = mes_mais_recente_v1["Patrimonio_Liquido"] / mes_mais_recente_v1["Cotas_Emitidas"]
            
            fig2 = go.Figure()
            
            fig2.add_trace(go.Scatter(
                x=mes_mais_recente_v1['Data_Referencia'],
                y=mes_mais_recente_v1['valor_patrimonial'],
                mode='lines+markers',
                name='Valor da Cota',
                line=dict(color='#10b981', width=2),
                marker=dict(size=8, symbol='circle')
            ))
            
            fig2.update_layout(
                title={
                    'text': 'Histórico do Valor Patrimonial',
                    'y':0.9,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                xaxis_title='Data',
                yaxis_title='Valor da Cota (R$)',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128,128,128,0.2)',
                    tickformat='%B/%Y'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128,128,128,0.2)',
                    tickprefix='R$ '
                ),
                hovermode='x unified',
                margin=dict(l=60, r=30, t=80, b=60)
            )
            
            # Adicionar informação de hover personalizada
            fig2.update_traces(
                hovertemplate="<b>Data:</b> %{x|%B/%Y}<br>" +
                            "<b>Valor:</b> R$ %{y:.2f}<br>"
            )
            
            # Adicionar os gráficos ao Streamlit
            st.plotly_chart(fig2, use_container_width=True)
            st.write("testeeeeeeee")

            ########################### LLM ##################
            st.markdown("### Análise do FII")
            
            with st.spinner("Gerando análise detalhada do FII..."):
                analysis = gerar_analise_fii(
                    st.session_state.ticker_escolhido,
                    round(dividend_yield, 2),
                    formatar_numero(patrimonio_liquido),
                    round(valor_cota, 2),
                    formatar_numero(cotistas),
                    segmento_atuacao
                )
                st.markdown(analysis)

    return st.session_state.ticker_escolhido

########################################################### APIS ##############################################################

app = FastAPI()


openai_client = OpenAI(api_key=api_key)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class ChatResponse(BaseModel):
    response: str
    messages: List[Message]

SYSTEM_CONTEXT = """Você é um especialista em Fundos Imobiliários (FIIs) do mercado brasileiro.
Forneça respostas claras e objetivas sobre FIIs, incluindo análises, recomendações e explicações
sobre conceitos importantes do mercado. Mantenha um tom profissional e educativo."""

@app.post("/chat/especialista_fii", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest):
    try:
        messages_for_api = [
            {"role": "system", "content": SYSTEM_CONTEXT}
        ] + [message.model_dump() for message in chat_request.messages]
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages_for_api,
            temperature=0.7,
            max_tokens=1000
        )
        
        assistant_response = response.choices[0].message.content
        updated_messages = chat_request.messages + [
            Message(role="assistant", content=assistant_response)
        ]
        
        return ChatResponse(
            response=assistant_response,
            messages=updated_messages
        )
        
    except ValidationError as e:
        raise HTTPException(status_code=422, detail="Erro de validação nos dados")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

# Função auxiliar para processar consultas de CNPJ
def process_cnpj_query(df: pd.DataFrame, cnpj: int, tipo_consulta: str):
    try:
        df = df.fillna(0)
        df['cnpj_normalizado'] = df['CNPJ_Fundo'].str.replace(".", "").str.replace("/", "").str.replace("-", "").astype(int)
        fundo = df[df['cnpj_normalizado'] == cnpj]
        
        if fundo.empty:
            raise HTTPException(
                status_code=404, 
                detail=f"CNPJ não encontrado na base de {tipo_consulta}"
            )
        
        return fundo.to_dict(orient="records")
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=500, 
            detail=f"Erro ao processar dados de {tipo_consulta}: {str(e)}"
        )

@app.get("/dataset/ativos_passivos/{cnpj}")
async def read_concatenacao(cnpj: int):
    """
    Endpoint destinado a obter as informações dos ativos e passivos desde 2020 ate o mês mais recente, por CNPJ desejado
    """
    try:
        df = concatenacao_at_pas()
        return process_cnpj_query(df, cnpj, "ativos e passivos")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Erro ao consultar ativos e passivos: {str(e)}"
        )

@app.get("/dataset/complemento/{cnpj}")
async def read_concatenacao(cnpj: int):
    """
    Endpoint destinado a obter as informações dos complementos desde 2020 ate o mês mais recente, por CNPJ desejado
    """
    try:
        df = concatenacao_complement()
        return process_cnpj_query(df, cnpj, "complementos")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Erro ao consultar complementos: {str(e)}"
        )

@app.get("/dataset/geral/{cnpj}")
async def read_concatenacao(cnpj: int):
    """
    Endpoint destinado a obter as informações gerais desde 2020 ate o mês mais recente, por CNPJ desejado
    """
    try:
        df = concatenacao_geral()
        return process_cnpj_query(df, cnpj, "informações gerais")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Erro ao consultar informações gerais: {str(e)}"
        )


########################################################### EXIBIÇÃO ##########################################################
def pagina_home():
    inicial()
    data()

def pagina_metricas():
    st.title("Análise exploratória dos dados")
    analise_exploratoria_v1()
    st.write("")
    metrica_at_pas_v1(concatenacao_at_pas)
    st.write("")
    segmento_fiis(concatenacao_geral)
    st.write("")
    scatter_plot(concatenacao_complement)
    st.write("")


def pagina_modelo_recomendacao():
    st.title("Modelo de recomendação")
    
   # Inicialização das variáveis de estado
    if 'etapa' not in st.session_state:
        st.session_state.etapa = 0
    if 'proximo_clicado' not in st.session_state:
        st.session_state.proximo_clicado = False
    
    # Função para controlar o avanço de etapa
    def avancar_etapa():
        st.session_state.proximo_clicado = True
        st.session_state.etapa += 1

    # Função para resetar o estado do botão próximo
    def reset_proximo():
        st.session_state.proximo_clicado = False

    # Etapa 1: Atualizar dados
    if st.session_state.etapa == 0:
        st.subheader("Etapa 1: Atualizando os dados")
        atualizar_dados_scrapping()
        st.write("Para avançar para a próxima etapa selecione a opção abaixo")
        if st.button("Próximo", key='proximo_1', on_click=reset_proximo):
            avancar_etapa()

    # Etapa 2: Escolha do perfil de investidor
    elif st.session_state.etapa == 1:
        st.subheader("Etapa 2: Escolha o perfil do investidor")
        investidor = recomendacao_inicial_investidor()
        if st.button("Próximo", key='proximo_2', on_click=reset_proximo):
            st.session_state.investidor = investidor
            avancar_etapa()

    # Etapa 3: Inicializando recomendação
    elif st.session_state.etapa == 2:
        st.subheader("Etapa 3: Inicializando recomendação")
        tipo_investidor = recomendacao_inicial()
        if st.button("Próximo", key='proximo_3', on_click=reset_proximo):
            st.session_state.tipo_investidor = tipo_investidor
            avancar_etapa()

    # Etapa 4: Valor disposto a investir
    elif st.session_state.etapa == 3:
        st.subheader("Etapa 4: Valor disposto a investir")
        valor = valor_disposto_investir()
        if st.button("Próximo", key='proximo_4', on_click=reset_proximo):
            st.session_state.valor_disposto = valor
            avancar_etapa()

    # Etapa 5: Escolha do tipo de histórico
    elif st.session_state.etapa == 4:
        st.subheader("Etapa 5: Escolha o tipo de histórico")
        historico = recomendacao_historica()
        if st.button("Próximo", key='proximo_5', on_click=reset_proximo):
            st.session_state.historico = historico
            avancar_etapa()

    # Etapa 6: Escolha do segmento
    elif st.session_state.etapa == 5:
        st.subheader("Etapa 6: Escolha o segmento")
        segmentos = recomendacao_segmento()
        if st.button("Próximo", key='proximo_6', on_click=reset_proximo):
            st.session_state.segmentos = segmentos
            avancar_etapa()

    # Etapa 7: Escolha da quantidade
    elif st.session_state.etapa == 6:
        st.subheader("Etapa 7: Escolha a quantidade")
        quantidade = recomendacao_quantidade()
        if st.button("Próximo", key='proximo_7', on_click=reset_proximo):
            st.session_state.quantidade = quantidade
            avancar_etapa()

    # Resumo das escolhas
    elif st.session_state.etapa == 7:
        st.subheader("Resumo das escolhas")
        st.write("Perfil do investidor:", st.session_state.get('investidor', 'Não selecionado'))
        st.write("Valor disposto a investir:", st.session_state.get('valor_disposto', 'Não selecionado'))
        st.write("Tipo de histórico:", st.session_state.get('historico', 'Não selecionado'))
        st.write("Segmentos selecionados:", st.session_state.get('segmentos', 'Não selecionado'))
        st.write("Quantidade escolhida:", st.session_state.get('quantidade', 'Não selecionado'))
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Reiniciar"):
                st.session_state.etapa = 0
                st.session_state.proximo_clicado = False
        
        with col2:
            if st.button("Finalizar"):
                st.session_state.etapa = 8  # Avança para a última etapa

    # Mostrando as recomendações
    elif st.session_state.etapa == 8:
        ticker_escolhido = escolha_analise()
        
    chat_fii()


def pagina_apis():
    st.title("Documentação API")
    endpoints_api()

def pagina_download():
    st.title("Download dos arquivos utilizados")
    st.write("Gostou do projeto? Fique a vontade para realizar download dos dados como preferir!")
    datasets_download(concatenacao_at_pas, concatenacao_complement, concatenacao_geral)


def Main():
    st.sidebar.title("Navegação")

    if "pagina_selecionada" not in st.session_state:
        st.session_state["pagina_selecionada"] = "Página Inicial"


    if st.sidebar.button("Página Inicial"):
        st.session_state["pagina_selecionada"] = "Página Inicial"
    if st.sidebar.button("Análise exploratória"):
        st.session_state["pagina_selecionada"] = "Análise exploratória"
    if st.sidebar.button("Modelo de recomendação"):
        st.session_state["pagina_selecionada"] = "Modelo de recomendação"
    if st.sidebar.button("Documentação API"):
        st.session_state["pagina_selecionada"] = "Documentação API"
    if st.sidebar.button("Download arquivos"):
        st.session_state["pagina_selecionada"] = "Download arquivos"

    # Navegação condicional com base no estado da sessão
    if st.session_state["pagina_selecionada"] == "Página Inicial":
        pagina_home()
    elif st.session_state["pagina_selecionada"] == "Análise exploratória":
        pagina_metricas()
    elif st.session_state["pagina_selecionada"] == "Modelo de recomendação":
        pagina_modelo_recomendacao()
    elif st.session_state["pagina_selecionada"] == "Documentação API":
        pagina_apis()
    elif st.session_state["pagina_selecionada"] == "Download arquivos":
        pagina_download()

if __name__ == "__main__":
    Main()