from langchain.agents import AgentExecutor
from dotenv import load_dotenv
from agente import AgenteOpenAIFunctions

# O dotenv é usado para carregar as variáveis de ambiente do arquivo .env
load_dotenv()

pergunta = "Quais os dados de Ana?"

agente = AgenteOpenAIFunctions()
executor = AgentExecutor(agent=agente.agente, tools=agente.tools, verbose=True)

resposta = executor.invoke({"input" : pergunta})
print(resposta)
