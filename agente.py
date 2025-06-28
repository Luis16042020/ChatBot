from estudante import DadosDeEstudantes
from langchain.agents import Tool
from langchain.agents import create_openai_tools_agent
from langchain_openai import ChatOpenAI
import os
from langchain import hub

class AgenteOpenAIFunctions:
    def __init__(self):
        llm = ChatOpenAI(model="gpt-4o",
                        api_key=os.getenv("OPENAI_API_KEY"))
        
        dados_de_estudante = DadosDeEstudantes()

        self.tools = [
            Tool(
                name=dados_de_estudante.name,
                func=dados_de_estudante.run,
                description=dados_de_estudante.description
            )
        ]

        prompt = hub.pull("hwchase17/openai-functions-agent")
        print(prompt)

        self.agente = create_openai_tools_agent(llm, self.tools, prompt)