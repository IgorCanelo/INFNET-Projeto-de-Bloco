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
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from PyPDF2 import PdfReader

load_dotenv('env/config.txt')
api_key = os.getenv('API_KEY')


def inicial():

    st.markdown("<h1 style='text-align: center;'>💰 Recomendação de FII's 💰</h1>", unsafe_allow_html=True)

    st.markdown("""
    <hr>
    <h2>🎯 Objetivos do Projeto</h2>
    <ul>
        <li><b>Desenvolver</b> um modelo de recomendação de Fundos de Investimento Imobiliário (FII's).</li>
        <li><b>Auxiliar</b> investidores com pouca ou nenhuma experiência.</li>
        <li><b>Melhorar</b> os ganhos financeiros e otimizar o tempo dos usuários.</li>
    </ul>
    <hr>
    """, unsafe_allow_html=True)

    st.markdown("""
    <h2>🌟 Inspiração</h2>
    <p>Para mais informações e inspiração, visite o <a href="https://www.fundsexplorer.com.br/ranking" target="_blank" style="color: #4CAF50; text-decoration: none;"><b>Ranking de FIIs do Funds Explorer</b></a>.</p>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)



def data():
    st.markdown(
        """
        <h2>📊 Amostra dos Dados Utilizados no Modelo</h2>
        """, unsafe_allow_html=True
    )

    st.markdown(
        """
        <p style='color: #333; font-size: 16px;'>
        Aqui está uma prévia do <b>dataset de 2024</b> que será utilizado para o modelo de recomendação de FIIs.
        </p>
        """, unsafe_allow_html=True
    )
    df_2024 = pd.read_csv("data/inf_mensal_fii_2024/inf_mensal_fii_complemento_2024.csv", delimiter=";", encoding="ISO-8859-1")
    st.dataframe(df_2024)



import streamlit as st

def analise_exploratoria_v1():

    st.markdown(
        """
        <h3>🔍 Obtenção dos Dados</h3>
        """, unsafe_allow_html=True
    )

    st.markdown(
        """
        <p style='font-size: 16px; color: #333; line-height: 1.6;'>
        Os dados utilizados são provenientes da <b>CVM</b>, organizados por ano. Para cada ano, temos três datasets:
        <ul>
            <li>Informações sobre <b>ativo vs passivo</b></li>
            <li>Complemento de dados</li>
            <li>Informações gerais dos fundos</li>
        </ul>
        </p>
        """, unsafe_allow_html=True
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown(
        """
        <h2>📂 Passo a passo para insights obtidos</h2>
        """, unsafe_allow_html=True
    )

    st.markdown(
        """
        <ul style='font-size: 16px; line-height: 1.8;'>
            <li>🛠️ <b>Concatenação:</b> Combinação dos datasets de todos os anos, resultando em apenas 3 datasets finais.</li>
            <li>📊 <b>Análise de ativos e passivos:</b> Visualizações para identificar quais fundos possuem mais direitos do que obrigações.</li>
            <li>🌎 <b>Segmento dos FII's:</b> Identificação dos fundos com maior atuação no Brasil.</li>
            <li>📈 <b>Dividend Yield:</b> Comparação anual dos dividendos distribuídos.</li>
        </ul>
        """, unsafe_allow_html=True
    )


def explicacao():

    st.markdown(
        """
        <h3>📘 Explicação de Termos Relacionados a FIIs</h3>
        """, unsafe_allow_html=True
    )

    st.markdown("""
    Este espaço é destinado a explicar as principais **nomenclaturas** normalmente utilizadas no contexto de **investimentos em fundos imobiliários (FIIs)**.  
    Vamos descomplicar os termos mais usados e auxiliar no entendimento do mercado!
    """)

    st.write("")
    st.write("")
    st.write("")

    st.subheader("🏢 Termos Relacionados aos Fundos Imobiliários (FIIs)")
    st.markdown("""
    - **Rendimentos**: Pagamentos periódicos feitos aos cotistas (similar aos dividendos).
    - **Dividend Yield (DY)**: Percentual do rendimento pago em relação ao preço da cota.
    - **Preço sobre Valor Patrimonial (P/VP)**: Relação entre o preço da cota e o valor patrimonial do fundo.
    - **Vacância**: Porcentagem de imóveis ou áreas que estão desocupadas no portfólio do FII.
    - **Cap Rate**: Taxa de capitalização, usada para avaliar a rentabilidade de um imóvel.
    - **Taxa de Administração**: Taxa paga ao gestor do fundo para sua administração.
    - **Gestor**: Profissional ou empresa responsável pela gestão do fundo.
    - **Ativo-alvo**: Tipo de imóvel ou investimento no qual o fundo aplica recursos (shoppings, galpões logísticos, lajes corporativas, etc.).
    - **FII de Papel**: Fundos que investem em títulos como CRIs (Certificados de Recebíveis Imobiliários).
    - **FII de Tijolo**: Fundos que investem diretamente em imóveis físicos.
    - **CRI (Certificado de Recebíveis Imobiliários)**: Título de renda fixa lastreado em créditos do setor imobiliário.
    """)

    st.write("")
    st.write("")
    st.write("")
    st.subheader("📊 Indicadores de Desempenho")
    st.markdown("""
    - **Ativos**: É tudo o que gera ou pode gerar valor para empresa, ou seja, recursos disponíveis.
    - **Passivos**: São as obrigações ou dívidas que precisam ser pagas no futuro, ou seja, são as obrigações da empresa.
    - **Liquidez**: Facilidade de comprar ou vender cotas no mercado.
    - **Rentabilidade**: Retorno obtido em relação ao investimento inicial.
    - **Volatilidade**: Medida de variação do preço das cotas ao longo do tempo.
    - **Patrimônio Líquido (PL)**: Valor total dos ativos do fundo menos os passivos.
    - **Resultado por Cota (R$/cota)**: Lucro distribuível dividido pelo número de cotas.
    - **TIR (Taxa Interna de Retorno)**: Mede a rentabilidade esperada do fundo.
    - **VPA (Valor Patrimonial por Ação)**: Valor patrimonial dividido pelo número de cotas.
    """)

    st.write("")
    st.write("")
    st.write("")
    st.subheader("💼 Termos Gerais do Mercado de Investimentos")
    st.markdown("""
    - **Renda Fixa**: Investimentos com retorno previsível, como títulos públicos ou CRIs.
    - **Renda Variável**: Investimentos cujo retorno não é garantido, como FIIs ou ações.
    - **Diversificação**: Estratégia para reduzir riscos alocando recursos em diferentes ativos.
    - **Índice de Referência (Benchmark)**: Indicador usado para medir o desempenho, como o IFIX para FIIs.
    - **IFIX**: Índice de Fundos Imobiliários da Bolsa Brasileira (B3).
    - **Alavancagem**: Uso de capital de terceiros para aumentar o potencial de retorno (ou risco).
    - **Tesouro Direto**: Programa de investimento em títulos públicos do governo.
    """)

    st.write("")
    st.write("")
    st.write("")
    st.subheader("📜 Termos Jurídicos e Tributários")
    st.markdown("""
    - **Isenção de IR**: Fundos imobiliários têm rendimentos isentos para pessoas físicas, desde que atendam a critérios legais.
    - **Proventos**: Distribuições financeiras aos cotistas (rendimento ou amortização).
    - **Amortização**: Devolução de parte do capital investido pelo cotista.
    - **Taxa de Custódia**: Cobrança pelo armazenamento das cotas em instituições financeiras.
    """)

    st.write("")
    st.write("")
    st.write("")
    st.subheader("📂 Tipos de Fundos Imobiliários")
    st.markdown("""
    - **Híbridos**: Fundos que combinam diferentes tipos de ativos (papéis + tijolos).
    - **Monoativo**: Fundos que possuem apenas um imóvel.
    - **Multimercado**: Fundos que diversificam investimentos em setores e regiões distintas.
    """)

    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.markdown("""
    🌟 **Agora você está preparado(a) para entender os termos mais usados no mundo dos FIIs!**
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

    
    if st.button("Atualizar Conjunto de Dados"):
        atualizar_dados()
        st.write("Conjunto de Dados Atualizado")
    

def recomendacao_inicial():
    st.markdown("""
Quais são os principais perfis de investidores?

- **Conservador** - Evita riscos e prefere investimentos com menor retorno e maior segurança.
- **Moderado** - Aceita mais riscos do que o conservador, mas volta a investir em opções mais seguras em momentos de instabilidade.
- **Arrojado** - Tem maior apetite ao risco e busca maiores retornos financeiros.
""")
    opcoes = ['Conservador', 'Moderado', 'Arrojado']
    perfil_investidor = st.selectbox("Selecione o seu perfil de investidor", opcoes)
    st.write(f'Você selecionou: {perfil_investidor}')
    st.session_state["investidor_conser_mod_arroj"] = perfil_investidor
    return perfil_investidor

def endpoints_api():
    st.markdown("<h2>🔌 Endpoints Utilizados para Obter Informações via API</h2>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True) 
    
    st.markdown("""
    ## Teste os Endpoints da API
    Para realizar testes e se familiarizar com os endpoints da API, recomendamos acessar o Swagger no endpoint abaixo:
    """)
    st.markdown("<h3 style='text-align: center; color: #28A745;'>🔗 Acessar Swagger - Testar Endpoints http://127.0.0.1:8000/docs#/</h3>", unsafe_allow_html=True)
    
    # Adicionando espaçamento
    st.markdown("<hr>", unsafe_allow_html=True) 
    
    # POST - Assistente Virtual
    st.subheader("🔌 **POST - Conversar com especialista virtual focado em FII's**")
    st.write("Este endpoint é utilizado para obter uma conversação com o assistente financeiro virtual para tirar dúvidas sobre FII's.")
    st.markdown("<h4 style='color: #FF5722;'>Endpoint:</h4>", unsafe_allow_html=True)
    st.markdown("<h5 style='color: #FF5722;'>http://127.0.0.1:8000/chat/especialista_fii</h5>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True) 

    # GET - Ativos e Passivos
    st.subheader("🔌 **GET - Obter as informações de ativos e passivos por CNPJ**")
    st.write("Esse endpoint retorna todas as informações de ativos e passivos do conjunto de dados desde 2020 até a data mais recente, com o CNPJ como parâmetro.")
    st.markdown("<h4 style='color: #FF5722;'>Endpoint:</h4>", unsafe_allow_html=True)
    st.markdown("<h5 style='color: #FF5722;'>http://127.0.0.1:8000/dataset/ativos_passivos/{cnpj}</h5>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True) 

    # GET - Complemento
    st.subheader("🔌 **GET - Obter as informações de complemento por CNPJ**")
    st.write("Esse endpoint retorna todas as informações de complemento do conjunto de dados desde 2020 até a data mais recente, com o CNPJ como parâmetro.")
    st.markdown("<h4 style='color: #FF5722;'>Endpoint:</h4>", unsafe_allow_html=True)
    st.markdown("<h5 style='color: #FF5722;'>http://127.0.0.1:8000/dataset/complemento/{cnpj}</h5>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True) 

    # GET - Informações Gerais
    st.subheader("🔌 **GET - Obter as informações gerais por CNPJ**")
    st.write("Esse endpoint retorna todas as informações gerais do conjunto de dados desde 2020 até a data mais recente, com o CNPJ como parâmetro.")
    st.markdown("<h4 style='color: #FF5722;'>Endpoint:</h4>", unsafe_allow_html=True)
    st.markdown("<h5 style='color: #FF5722;'>http://127.0.0.1:8000/dataset/geral/{cnpj}</h5>", unsafe_allow_html=True)
    



def recomendacao_inicial_investidor():
    st.markdown("""
- **Iniciante** - Tenho pouco ou nenhum conhecimento sobre investimentos em FIIs.
- **Intermediário** - Já investi em FIIs, mas ainda tenho poucos conhecimentos sobre o tema.
- **Avançado** - Conheço o mercado de FIIs, possuo capital alocado e acompanho o desempenho regularmente.
""")
    opcoes = ['Iniciante', 'Intermediário', 'Avançado']
    investidor = st.selectbox("Selecione o seu nível de conhecimento em FIIs", opcoes)
    st.write(f'Você selecionou: {investidor}')
    st.session_state["investidor_inic_med_avan"] = investidor
    return investidor

def valor_disposto_investir():

    st.markdown("""
Qual o valor que você tem em mente para investir por cota?
""")

    valor_investido = st.selectbox(
        "Selecione as opções:",
        ["R$0,00 a R$90,00", "R$91,00 a R$120,00", "Acima de R$121,00", "Não possuo valor pré-estabelecido"]
    )
    st.session_state["valor_investir"] = valor_investido
    return valor_investido

def recomendacao_historica():

    st.markdown("""
Você deseja obter essa recomendação com base em:

- **Histórica** - Desde a criação do FII.  
- **Anual** - O ano atual.
- **Mensal** - O mês atual.    
""")
    opcoes = ['Histórica', 'Anual', 'Mensal']
    historico = st.radio("Selecione o período desejado para obter a recomendação", opcoes)
    st.write(f'Você selecionou: {historico}')
    st.session_state["historico"] = historico 
    return historico


def recomendacao_segmento():
    st.markdown("""
Selecione os segmentos dos FIIs a serem recomendados:
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
Selecione quantos FIIs deseja obter de recomendação:
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
    perfil_investidor = st.session_state.get('investidor_conser_mod_arroj')
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
    valor_disposto = st.session_state.get('valor_investir')
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
    

def llm_resumo(lista):
    openai.api_key = api_key

    prompt = f"""Você irá analisar o conteúdo a seguir, que consiste em um relatório gerencial de um fundo imobiliário, e deverá resumir as informações mais relevantes de forma clara e objetiva, sem usar listas. O objetivo é destacar os pontos principais, incluindo:
    1. Os **objetivos estratégicos** do fundo, como metas de rentabilidade, diversificação e crescimento.
    2. **Resultados financeiros atualizados** do mês, como rentabilidade, dividendos pagos e variação do patrimônio líquido.
    3. **Indicadores de performance** importantes, como o rendimento por cota, a valorização das cotas e o comparativo com benchmarks do mercado.
    4. **Principais investimentos e ativos do fundo**, incluindo a performance desses ativos no mês e quaisquer mudanças significativas na carteira.
    5. **Gestão de riscos**: aspectos como a alavancagem utilizada, a exposição a diferentes setores e a diversificação geográfica ou de ativos.
    6. **Perspectivas futuras** do fundo, incluindo estratégias planejadas para os próximos meses ou anos.
    O resumo deve ser fluido, sem perder a objetividade, e cobrir todos esses pontos de forma integrada, destacando as informações mais relevantes para os investidores e stakeholders, sem simplesmente repetir os dados da lista. A lista com as informações a serem resumidas é a seguinte: {lista}
"""
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erro ao gerar análise: {str(e)}"





def gerar_analise_fii(ticker, dividend_yield, patrimonio_liquido, valor_cota, cotistas, segmento):
    openai.api_key = api_key
    
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
    
    st.markdown("### Com base nas suas escolhas, as recomendações são:")

    df = score_df()

    df_complemento = concatenacao_complement()
    df_complemento['Data_Referencia'] = pd.to_datetime(df_complemento['Data_Referencia'], dayfirst=True, format='mixed')
    
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
                    <div class="metric-label">Dividend Yield Patrimonial</div>
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
            
            
            fig2.update_traces(
                hovertemplate="<b>Data:</b> %{x|%B/%Y}<br>" +
                            "<b>Valor:</b> R$ %{y:.2f}<br>"
            )
            
            st.plotly_chart(fig2, use_container_width=True)

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
                st.write("")
                st.write("")
                st.markdown("<hr>", unsafe_allow_html=True)
                st.write("")
                st.write("")
                st.subheader("Resumo do relatório gerencial mais recente disponível")
                ticker = str(st.session_state.ticker_escolhido).lower()  # Código do fundo imobiliário
                diretorio_download = "data/downloads"
                headless = True  # Executar navegador em modo headless
                relatorio_textos = relatorio_gerencial(ticker, diretorio_download, headless)
                st.write(llm_resumo(relatorio_textos))

    st.markdown("<h3 style='text-align: center;'>🤖 Tire suas dúvidas com o Assistente Virtual!</h3>", unsafe_allow_html=True)
    chat_fii()

    return st.session_state.ticker_escolhido

def scrapping_relatorio(ticker):

    url = f"https://www.fundsexplorer.com.br/funds/{ticker}"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')

    links = soup.find_all('a', href=True)

    data_atual = datetime.now()
    ano = data_atual.year
    mes = data_atual.month

    def retroceder_mes(ano, mes):
        """Retrocede um mês, ajustando o ano se necessário."""
        if mes == 1:
            mes = 12
            ano -= 1
        else:
            mes -= 1
        return ano, mes

    relatorio_mais_recente = None
    while not relatorio_mais_recente:
        mes_ano = f"{mes:02d}/{ano}"

        for link in links:
            texto = link.get_text(strip=True).lower()
            if 'gerencial' in texto and mes_ano in texto:
                relatorio_mais_recente = link['href']
                print(f"Relatório encontrado para: {mes_ano}")
                break

        ano, mes = retroceder_mes(ano, mes)
    return relatorio_mais_recente

def configurar_download_automatico(diretorio_download, headless=False):
    chrome_options = Options()
    
    # Configurações para download automático
    chrome_options.add_experimental_option('prefs', {
        "download.default_directory": diretorio_download,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "plugins.always_open_pdf_externally": True,
        "safebrowsing.enabled": True
    })
    
    if headless:
        chrome_options.add_argument('--headless=new')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--window-size=1920,1080')
    
    return chrome_options

def baixar_pdf_selenium(url, diretorio_download, headless=False):

    diretorio_download = os.path.abspath(diretorio_download)
    
    if not os.path.exists(diretorio_download):
        os.makedirs(diretorio_download)
    
    chrome_options = configurar_download_automatico(diretorio_download, headless)
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), 
                            options=chrome_options)
    
    try:
        driver.get(url)
        wait = WebDriverWait(driver, 10)
        botao_download = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="icon"]/cr-icon')))
        botao_download.click()
        time.sleep(5)
        print(f"Download iniciado. Verifique a pasta: {diretorio_download}")
        
    except Exception as e:
        print(f"Erro ao fazer download: {e}")
    
    finally:
        driver.quit()


def ult():

    downloads_path = "data/downloads"
    arquivos = os.listdir(downloads_path)

    lista = []

    for arquivo in arquivos:
        caminho_arquivo = os.path.join(downloads_path, arquivo)
        if arquivo.lower().endswith(".pdf"):
            try:
                reader = PdfReader(caminho_arquivo)
                for i, page in enumerate(reader.pages):
                    print(f"--- Página {i + 1} do arquivo {arquivo} ---")
                    texto = page.extract_text()
                    lista.append(texto)
            except Exception as e:
                print(f"Erro ao processar o arquivo {arquivo}: {e}")

    for arquivo in arquivos:
        caminho_arquivo = os.path.join(downloads_path, arquivo)
        try:
            os.remove(caminho_arquivo)
            print(f"Arquivo {arquivo} removido com sucesso.")
        except Exception as e:
            print(f"Erro ao excluir o arquivo {arquivo}: {e}")

    return lista


def relatorio_gerencial(ticker, diretorio_download, headless=False):
    """
    Função principal que integra as etapas de scraping, download e processamento
    dos relatórios gerenciais de um fundo imobiliário.
    
    Args:
        ticker (str): Código do fundo imobiliário.
        diretorio_download (str): Diretório onde os relatórios serão baixados.
        headless (bool): Se True, executa o navegador em modo headless.
        
    Returns:
        list: Lista de textos extraídos dos PDFs baixados.
    """
    try:
        print("Iniciando scraping para encontrar o relatório...")
        url_relatorio = scrapping_relatorio(ticker)
        if not url_relatorio:
            print("Nenhum relatório encontrado.")
            return None
        
        print(f"Link do relatório encontrado: {url_relatorio}")
        
        print("Iniciando download do relatório...")
        baixar_pdf_selenium(url_relatorio, diretorio_download, headless)
        
        print("Processando PDFs baixados...")
        textos_extraidos = ult()
        
        print("Processamento concluído.")
        return textos_extraidos

    except Exception as e:
        print(f"Erro durante a execução: {e}")
        return None



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
    st.markdown("<h1 style='text-align: center;'>📊 Explorando os Dados para Insights</h1>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    analise_exploratoria_v1()
    st.write("")
    st.markdown("<hr>", unsafe_allow_html=True)
    explicacao()
    st.write("")
    st.markdown("<hr>", unsafe_allow_html=True)
    metrica_at_pas_v1(concatenacao_at_pas)
    st.write("")
    st.markdown("<hr>", unsafe_allow_html=True)
    segmento_fiis(concatenacao_geral)
    st.write("")
    st.markdown("<hr>", unsafe_allow_html=True)
    scatter_plot(concatenacao_complement)
    st.write("")


def pagina_modelo_recomendacao():

    if 'etapa' not in st.session_state:
        st.session_state.etapa = 0


    def avancar_para_resumo():
        st.session_state.etapa = 1

    def finalizar():
        st.session_state.etapa = 2


    st.markdown("<h1 style='text-align: center;'>💰 Modelo de Recomendação para Fundos Imobiliários 💰</h1>", unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.markdown("<hr>", unsafe_allow_html=True) 
    st.write("")
    st.write("")


    if st.session_state.etapa == 0:
        # Etapa 1
        with st.container():
            st.markdown("## 🛠️ Etapa 1: Atualizando os dados")
            st.info("Clique no botão abaixo para atualizar todo o conjunto de dados do modelo.")
            atualizar_dados_scrapping()
        st.write("")
        st.write("")
        st.markdown("<hr>", unsafe_allow_html=True) 
        st.write("")
        st.write("")

        # Etapa 2
        with st.container():
            st.markdown("## 👤 Etapa 2: Escolha o perfil do investidor")
            st.markdown("**Como você se classifica em relação a investimento em Fundos Imobiliários (FIIs)?**")
            recomendacao_inicial_investidor()
        st.write("")
        st.write("")
        st.markdown("<hr>", unsafe_allow_html=True) 
        st.write("")
        st.write("")

        # Etapa 3
        with st.container():
            st.markdown("## 📊 Etapa 3: Inicializando recomendação")
            recomendacao_inicial()
        st.write("")
        st.write("")
        st.markdown("<hr>", unsafe_allow_html=True) 
        st.write("")
        st.write("")

        # Etapa 4
        with st.container():
            st.markdown("## 💵 Etapa 4: Valor disposto a investir")
            valor_disposto_investir()
        st.write("")
        st.write("")
        st.markdown("<hr>", unsafe_allow_html=True) 
        st.write("")
        st.write("")

        # Etapa 5
        with st.container():
            st.markdown("## 🕒 Etapa 5: Escolha o tipo de histórico")
            recomendacao_historica()
        st.write("")
        st.write("")
        st.markdown("<hr>", unsafe_allow_html=True) 
        st.write("")
        st.write("")

        # Etapa 6
        with st.container():
            st.markdown("## 🏢 Etapa 6: Escolha o segmento")
            recomendacao_segmento()
        st.write("")
        st.write("")
        st.markdown("<hr>", unsafe_allow_html=True) 
        st.write("")
        st.write("")

        # Etapa 7
        with st.container():
            st.markdown("## 🔢 Etapa 7: Escolha a quantidade")
            recomendacao_quantidade()
        st.write("")
        st.write("")
        st.markdown("<hr>", unsafe_allow_html=True) 
        st.write("")
        st.write("")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Próximo ➡️"):
                avancar_para_resumo()

    # Resumo das escolhas (etapa 1)
    elif st.session_state.etapa == 1:
        st.markdown("## 📝 Resumo das Escolhas:")
        st.write("")
        st.write("### 👤 **Perfil do investidor:**", st.session_state.get('investidor_inic_med_avan', 'Não selecionado'))
        st.write("")
        st.write("### 📊 **Tipo de investidor:**", st.session_state.get('investidor_conser_mod_arroj', 'Não selecionado'))
        st.write("")
        st.write("### 💵 **Valor disposto a investir:**", st.session_state.get('valor_investir', 'Não selecionado'))
        st.write("")
        st.write("### 🕒 **Tipo de histórico:**", st.session_state.get('historico', 'Não selecionado'))
        st.write("")
        st.write("### 🏢 **Segmentos selecionados:**", st.session_state.get('segmentos', 'Não selecionado'))
        st.write("")
        st.write("### 🔢 **Quantidade escolhida:**", st.session_state.get('quantidade', 'Não selecionado'))
        st.write("")
        st.success("Confira as escolhas feitas antes de finalizar")
        st.write("")
        st.write("")
        st.write("")
        st.write("")


        col1, col2, col3 = st.columns([1, 2, 1])  # Centraliza o botão
        with col2:
            if st.button("Finalizar ✅"):
                finalizar()

    # Etapa final
    elif st.session_state.etapa == 2:
        st.markdown("## 🎉 Etapa Final: Conclusão")
        escolha_analise()

        


def pagina_apis():
    st.markdown("<h1 style='text-align: center;'> 🌐 Documentação API </h1>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    endpoints_api()

def pagina_download():
    st.markdown("<h1 style='text-align: center;'> 💾 Download dos arquivos utilizados </h1>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    ### Gostou do projeto?  
    📥 **Baixe os dados** utilizados no modelo e explore as informações como preferir!
    """)
    st.markdown("<hr>", unsafe_allow_html=True)
    datasets_download(concatenacao_at_pas, concatenacao_complement, concatenacao_geral)


def Main():
    st.sidebar.title("Navegação")

    if "pagina_selecionada" not in st.session_state:
        st.session_state["pagina_selecionada"] = "Página Inicial"


    if st.sidebar.button("Página Inicial"):
        st.session_state["pagina_selecionada"] = "Página Inicial"
    if st.sidebar.button("Insights"):
        st.session_state["pagina_selecionada"] = "Insights"
    if st.sidebar.button("Modelo de recomendação"):
        st.session_state["pagina_selecionada"] = "Modelo de recomendação"
    if st.sidebar.button("Documentação API"):
        st.session_state["pagina_selecionada"] = "Documentação API"
    if st.sidebar.button("Download arquivos"):
        st.session_state["pagina_selecionada"] = "Download arquivos"

    # Navegação condicional com base no estado da sessão
    if st.session_state["pagina_selecionada"] == "Página Inicial":
        pagina_home()
    elif st.session_state["pagina_selecionada"] == "Insights":
        pagina_metricas()
    elif st.session_state["pagina_selecionada"] == "Modelo de recomendação":
        pagina_modelo_recomendacao()
    elif st.session_state["pagina_selecionada"] == "Documentação API":
        pagina_apis()
    elif st.session_state["pagina_selecionada"] == "Download arquivos":
        pagina_download()

if __name__ == "__main__":
    Main()