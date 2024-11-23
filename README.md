Project Charter: Recomendação de Fundos Imobiliários FII's

1. Escopo do Projeto

    O projeto visa desenvolver uma aplicação que recomenda fundos imobiliários (FIIs) adequados para investidores brasileiros com pouca ou nenhuma experiência em investimentos. A aplicação utilizará um modelo de machine learning para analisar dados de FIIs e perfis de investidores, fornecendo recomendações personalizadas que maximizem o retorno sobre investimento (ROI) e otimizem o tempo gasto na tomada de decisões financeiras.

2. Objetivos do Projeto

    Melhorar os ganhos financeiros dos investidores iniciantes ao oferecer recomendações precisas de FIIs que correspondam ao seu perfil de risco e objetivos financeiros.
    Otimizar o processo de investimento, tornando mais rápido e fácil para os usuários escolherem os FIIs mais adequados às suas necessidades.
    Educar financeiramente os investidores, proporcionando insights sobre os FIIs recomendados e ajudando-os a entender melhor o mercado financeiro.
    Aumentar a satisfação dos usuários por meio de uma interface intuitiva e um suporte contínuo, garantindo uma experiência positiva na utilização da aplicação.

3. Stakeholders do Projeto

    Investidores Iniciantes: Usuários finais que se beneficiarão das recomendações personalizadas para melhorar seus retornos financeiros.
    Equipe de Ciência de Dados: Responsável pelo desenvolvimento do modelo de recomendação, coleta e análise de dados, e manutenção do sistema.
    Desenvolvedores de Software: Encarregados de criar a aplicação, integrando o modelo de recomendação, e garantindo que a interface seja intuitiva e responsiva.
    Gerentes de Projeto: Coordenam todas as atividades, garantindo que o projeto seja entregue dentro do prazo, orçamento e escopo definidos.
    Consultores Financeiros: Podem fornecer expertise para assegurar que as recomendações de FIIs sejam financeiramente sólidas e adequadas ao público-alvo.
    Usuários Beta/Testers: Primeiros usuários que testarão a aplicação e fornecerão feedbacks para melhorias antes do lançamento oficial.
    Apoio Executivo: Patrocinadores ou líderes organizacionais que aprovam e financiam o projeto, assegurando o alinhamento com os objetivos estratégicos da organização.

4. Como realizar a Recomendação de Fundos Imobiliários FII's

    4.1.
        O primeiro passo foi a automatização da coleta de dados através de WebScraping na página da CVM. Esse processo foi implementado para garantir que os dados mais recentes estejam disponíveis para o modelo de recomendação. A grande vantagem dessa abordagem é a interatividade proporcionada ao usuário, que pode atualizar os dados sempre que desejar. Isso garante que o modelo seja alimentado com as informações mais atuais, maximizando a precisão das recomendações. A integração do WebScraping permite que o usuário tenha controle sobre o momento da coleta, evitando que o modelo se baseie em dados desatualizados e assegurando um fluxo contínuo e preciso de informações.

    4.2. 
        O segundo passo na recomendação de Fundos Imobiliários (FIIs) é realizar uma análise exploratória abrangente dos dados disponíveis. Essa etapa é fundamental para entender as características e nuances do conjunto de dados.

        Compreensão do Conjunto de Dados: 
            Embora os dados já estejam estruturados, é essencial dedicar tempo para explorá-los. Isso envolve examinar a distribuição dos dados, identificar variáveis relevantes e entender a relação entre elas.

        Limpeza e Preparação dos Dados: 
            Durante a análise, você pode encontrar inconsistências, valores ausentes ou outliers. A limpeza dos dados é crucial para garantir a qualidade das análises subsequentes. Isso pode envolver a remoção de duplicatas, o tratamento de valores ausentes e a normalização de variáveis.

        Visualização dos Dados: 
            A visualização é crucial para identificar tendências e padrões. Gráficos, como histogramas, gráficos de dispersão, podem ajudar a revelar informações importantes que não são imediatamente óbvias em tabelas de dados. Essa abordagem facilita a identificação de comportamentos atípicos e a compreensão das relações entre variáveis.

        Extração de Insights: 
            Além de simplesmente explorar os dados, é importante extrair insights que possam guiar o desenvolvimento do modelo de recomendação. Perguntas como "Quais FIIs apresentaram o melhor desempenho nos últimos anos?" ou "Quais características estão associadas aos FIIs de maior sucesso?" devem ser investigadas.

        Com uma compreensão sólida e insights extraídos dos dados, poderá avançar para as próximas etapas do processo de recomendação, que incluem o desenvolvimento e a validação do modelo.

    4.3.
        O terceiro passo foi o desenvolvimento de rotas de API para fornecer acesso programático aos dados coletados e processados. Essa etapa foi essencial para tornar os dados disponíveis de maneira flexível, permitindo que outros desenvolvedores e sistemas externos possam integrá-los conforme suas próprias necessidades. Com as rotas de API, os dados podem ser consumidos em diferentes formatos e integrados a diversas aplicações, seja para fins analíticos, de desenvolvimento de novas funcionalidades ou mesmo para personalizar o uso dos dados conforme o interesse de quem os acessa. Essa abordagem amplia o alcance do projeto, transformando-o em uma ferramenta versátil e adaptável a diferentes cenários e plataformas.

    4.4.
        O quarto passo envolve a utilização dos dados coletados e disponibilizados. A recomendação utiliza técnicas de socre para identificar padrões e gerar recomendações baseadas nos dados históricos e no comportamento atual do mercado. Esse modelo é projetado para recomendar os melhores FIIs (Fundos Imobiliários) com base em critérios como desempenho passado, características de risco, e outros fatores relevantes. Ao utilizar os dados mais recentes e bem estruturados, o modelo consegue oferecer recomendações personalizadas e adaptadas ao perfil do investidor, potencializando os resultados e orientando as melhores decisões de investimento.

    4.5. 
        O quinto passo, após calcular os scores dos fundos imobiliários (FIIs) com base nos critérios previamente estabelecidos, é desenvolver uma aplicação gráfica voltada para o usuário. Essa aplicação deve permitir a visualização dos FIIs de forma clara e intuitiva, oferecendo diversas opções de filtros e ferramentas de personalização. Dessa forma, o usuário poderá selecionar os investimentos que melhor atendam ao seu perfil, considerando fatores como estilo de investimento, capital disponível e outros critérios relevantes para sua tomada de decisão.
    
    4.6.
        O sexto passo consiste em integrar a LLM da OpenAI por meio de sua API. Essa integração permitirá que a aplicação gere um resumo detalhado das escolhas feitas pelo usuário, proporcionando uma visão clara e personalizada da decisão tomada. Além disso, durante todo o processo de seleção e recomendação, o usuário terá acesso a um chat interativo com a LLM, chamada de "Especialista Virtual". Essa funcionalidade oferecerá suporte para esclarecer dúvidas relacionadas a FIIs, tornando a experiência mais completa e informativa. A escolha de integrar a LLM da OpenAI via API foi motivada pelas limitações computacionais que enfrento atualmente. Essa abordagem permite utilizar o poder da LLM sem a necessidade de infraestrutura robusta ou recursos computacionais elevados. Além disso, a solução oferece resultados altamente satisfatórios, com respostas precisas e relevantes geradas pela LLM, atendendo plenamente às necessidades do projeto.



Data Summary Report: Recomendação de Fundos Imobiliários FII's

1. Fontes de Dados:

    Comissão de Valores Mobiliários (CVM):
        Tipo de Dados: Dados financeiros e regulatórios de fundos imobiliários, incluindo informações sobre rendimentos, volatilidade, liquidez, perfil de risco, patrimônio líquido, e histórico de desempenho.
        Objetivo de Uso: Os dados da CVM serão utilizados para alimentar o modelo de recomendação com informações precisas e atualizadas sobre os fundos imobiliários disponíveis no mercado. Esses dados ajudarão a criar perfis detalhados dos FIIs, que serão comparados com os perfis dos investidores para gerar recomendações personalizadas.

2. Estrutura de Armazenamento

    Pasta 'data':
        Todos os dados obtidos da CVM serão armazenados na pasta "data" dentro do ambiente de desenvolvimento do projeto.

3. Objetivo do Data Summary Report

    Este Data Summary Report servirá como um documento vivo que relaciona todas as fontes de dados utilizadas no projeto, permitindo que a equipe de ciência de dados e desenvolvimento tenha um entendimento claro das origens, tipos, e usos dos dados. Ele também facilita a rastreabilidade e governança dos dados, assegurando que todos os dados sejam adequadamente gerenciados e utilizados de forma consistente ao longo do projeto.
