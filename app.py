import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import time
import os
import requests
import shutil
from bs4 import BeautifulSoup
from zipfile import ZipFile
from fastapi import FastAPI

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


def endpoints_api():
    st.subheader("Endpoints utilizados para obter informações via API")
    st.write("Para realizar testes e se familiarizar com os endpoints da API recomendamos acessar o Swagger no endpoint abaixo:")
    st.write("http://127.0.0.1:8000/docs#/")
    st.subheader("GET - Obter as informações de ativos e passivos por CNPJ")
    st.write("Esse endpoint trás todas as informações de ativos e passivos do conjunto de dados desde 2020 ate a data mais recente, via CNPJ como parâmetro da requisição com apenas números.")
    st.write("http://127.0.0.1:8000/dataset/ativos_passivos/{cnpj}")
    st.subheader("GET - Obter as informações de complemento por CNPJ")
    st.write("Esse endpoint trás todas as informações de complemento do conjunto de dados desde 2020 ate a data mais recente, via CNPJ como parâmetro da requisição com apenas números.")
    st.write("http://127.0.0.1:8000/dataset/complemento/{cnpj}")
    st.subheader("GET - Obter as informações gerais por CNPJ")
    st.write("Esse endpoint trás todas as informações gerais do conjunto de dados desde 2020 ate a data mais recente, via CNPJ como parâmetro da requisição com apenas números.")
    st.write("http://127.0.0.1:8000/dataset/geral/{cnpj}")







########################################################### APIS ##############################################################
app = FastAPI()

@app.get("/dataset/ativos_passivos/{cnpj}")
async def read_concatenacao(cnpj: int):
    """
    Endpoint destinado a obter as informações dos ativos e passivos desde 2020 ate o mês mais recente, por CNPJ desejado
    """

    df = concatenacao_at_pas()
    df = df.fillna(0)
    df['cnpj_normalizado'] = df['CNPJ_Fundo'].str.replace(".", "").str.replace("/", "").str.replace("-", "").astype(int)
    fundo = df[df['cnpj_normalizado'] == cnpj]
    
    if fundo.empty:
        return {"error": "CNPJ não encontrado"}
    
    return fundo.to_dict(orient="records")

@app.get("/dataset/complemento/{cnpj}")
async def read_concatenacao(cnpj: int):
    """
    Endpoint destinado a obter as informações dos complementos desde 2020 ate o mês mais recente, por CNPJ desejado
    """

    df = concatenacao_complement()
    df = df.fillna(0)
    df['cnpj_normalizado'] = df['CNPJ_Fundo'].str.replace(".", "").str.replace("/", "").str.replace("-", "").astype(int)
    fundo = df[df['cnpj_normalizado'] == cnpj]
    
    if fundo.empty:
        return {"error": "CNPJ não encontrado"}
    
    return fundo.to_dict(orient="records")

@app.get("/dataset/geral/{cnpj}")
async def read_concatenacao(cnpj: int):
    """
    Endpoint destinado a obter as informações gerais desde 2020 ate o mês mais recente, por CNPJ desejado
    """

    df = concatenacao_geral()
    df = df.fillna(0)
    df['cnpj_normalizado'] = df['CNPJ_Fundo'].str.replace(".", "").str.replace("/", "").str.replace("-", "").astype(int)
    fundo = df[df['cnpj_normalizado'] == cnpj]
    
    if fundo.empty:
        return {"error": "CNPJ não encontrado"}
    
    return fundo.to_dict(orient="records")


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
    atualizar_dados_scrapping()
    recomendacao_inicial()



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