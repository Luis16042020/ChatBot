from estudante import DadosDeEstudantes, PerfilAcademico
from langchain.agents import Tool
from langchain.agents import create_openai_tools_agent, create_react_agent
from langchain_openai import ChatOpenAI
import os
from langchain import hub

class AgenteOpenAIFunctions:
    def __init__(self):
        llm = ChatOpenAI(model="gpt-4o",
                        api_key=os.getenv("OPENAI_API_KEY"))
        
        dados_de_estudante = DadosDeEstudantes()
        perfil_academico = PerfilAcademico()        

        self.tools = [
            Tool(
                name=dados_de_estudante.name,
                func=dados_de_estudante.run,
                description=dados_de_estudante.description
            ),
            Tool(
                name=perfil_academico.name,
                func=perfil_academico.run,
                description=perfil_academico.description
            )
        ]

        #Openai functions
        # prompt = hub.pull("hwchase17/openai-functions-agent")
        # self.agente = create_openai_tools_agent(llm, self.tools, prompt)

        #React
        prompt = hub.pull("hwchase17/react")
        self.agente = create_react_agent(llm, self.tools, prompt)
        print(prompt)
