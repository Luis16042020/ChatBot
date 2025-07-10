from langchain.tools import BaseTool
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
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
            # print(resposta)
            estudante = resposta['estudante']
            estudante = estudante.lower()
            dados = busca_dados_de_estudante(estudante)
            # print(dados)
            return json.dumps(dados)
        except Exception as e:
            print(f"Erro: {e}")

def busca_dados_de_estudante(estudante):
        dados = pd.read_csv(ESTUDANTES)
        dados_com_este_estudante = dados[dados["USUARIO"] == estudante]
        if dados_com_este_estudante.empty:
            return {}
        return dados_com_este_estudante.iloc[:1].to_dict()

class ExtratorDeEstudante(BaseModel):
    estudante: str = Field("Nome do estudante informado, sempre em letras minúsculas. Exemplo: joão, carlos, joana, carla")

class Nota(BaseModel):
    area:str = Field("Nome da área de conhecimento")
    nota:float = Field("Nota na área de conhecimento")

class PerfilAcademicoDeEstudante(BaseModel):
    nome:str = Field("nome do estudante")
    ano_de_conclusao:int = Field("ano de conclusão")
    notas:list[Nota] = Field("Lista de notas das disciplinas e áreas de conhecimento")
    resumo:str = Field("Resumo das principais características desse estudante de forma a torná-lo único e um ótimo potencial estudante para faculdades. Exemplo: só este estudante tem bla bla bla")

class PerfilAcademico(BaseTool):
     name: str = Field("PerfilAcademico")
     description: str = Field("Cria um perfil acadêmico de um estudante. Esta Ferramenta requer como entrada todos os dados do estudante.")

     def _run(self, input: str) -> str:
        llm = ChatOpenAI(model="gpt-4o", 
                            api_key=os.getenv("OPENAI_API_KEY"))
        parser = JsonOutputParser(pydantic_object=PerfilAcademicoDeEstudante)
        try:
            template = PromptTemplate(template="""- Formate o estudante para seu perfil acadêmico. Com os dados, 
                                        identifique as opções de universidades sugeridas e cursos compatíveis 
                                        com o interesse do aluno, destaque o perfil do aluno, dando enfase principalmente
                                        naquilo que faz sentido para as instituições de interesse do aluno
                                      
                                        Persona: você é uma consultora de carreira e precisa indicar com detalhes, riqueza, mas direta ao ponto 
                                        para o estudanteas opções, e consequências possíveis.
                                      
                                      
                                        Informações atuais:
                                        
                                        {dados_do_estudante}
                                        {formato_de_saida}
                                        """, input_variables=["dados_do_estudante"],
                                             partial_variables={"formato_de_saida" : parser.get_format_instructions()})
            cadeia = template | llm | parser
            resposta = cadeia.invoke({"dados_do_estudante" : input})
            return resposta

            
        except Exception as e:
            print(f"Erro: {e}")
            


