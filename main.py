from langchain.agents import AgentExecutor
from dotenv import load_dotenv
from langchain.tools import BaseTool
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain.agents import Tool
from langchain import hub
from langchain.agents import create_openai_tools_agent
import pandas as pd
from config.settings import ESTUDANTES
import json

class DadosDeEstudantes(BaseTool):
    name: str = "DadosDeEstudante"
    description: str = """" Esta ferramenta extrai o histórico e preferências de um estudante de acordo com seu histórico """

    def _run(self, input: str) -> str:
        try:
            llm = ChatOpenAI(model="gpt-4o", 
                            api_key=os.getenv("OPENAI_API_KEY"))
            parser = JsonOutputParser(pydantic_object=ExtratorDeEstudante)
            template = PromptTemplate(template="""Você deve analisar a {input} para extrair o nome de usuário informado.
            Formato de saida:
            {formato_saida}""",
            input_variables=["input"],
            partial_variables={"formato_saida": parser.get_format_instructions()})
            cadeia = template | llm | parser
            resposta = cadeia.invoke({"input" : input})
            print(resposta)
            estudante = resposta['estudante']
            dados = busca_dados_de_estudante(estudante)
            print(dados)
            return json.dumps
        except Exception as e:
            print(f"Erro: {e}")


def busca_dados_de_estudante(estudante):
        dados = pd.read_csv(ESTUDANTES)
        dados_com_este_estudante = dados[dados["USUARIO"] == estudante]
        if dados_com_este_estudante.empty:
            return {}
        return dados_com_este_estudante.iloc[:1].to_dict()

# O dotenv é usado para carregar as variáveis de ambiente do arquivo .env
load_dotenv()

class ExtratorDeEstudante(BaseModel):
    estudante: str = Field("Nome do estudante informado, sempre em letras minúsculas. Exemplo: joão, carlos, joana, carla")

pergunta = "Quais os dados de Ana?"

llm = ChatOpenAI(model="gpt-4o",
                        api_key=os.getenv("OPENAI_API_KEY"))

dados_de_estudante = DadosDeEstudantes()

tools = [
    Tool(
        name=dados_de_estudante.name,
        func=dados_de_estudante.run,
        description=dados_de_estudante.description
    )
]

prompt = hub.pull("hwchase17/openai-functions-agent")
print(prompt)

agente = create_openai_tools_agent(llm, tools, prompt)

executor = AgentExecutor(agent=agente, tools=tools, verbose=True)

resposta = executor.invoke({"input" : pergunta})
print(resposta)
# print(resposta)
