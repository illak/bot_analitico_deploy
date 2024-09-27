import os
import yaml
import pandas as pd
import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents.agent_types import AgentType
from typing import Annotated
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from IPython.display import display, Markdown
from langchain_core.output_parsers import StrOutputParser
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from fastapi import FastAPI, Depends, HTTPException, status, Query
from fastapi.security.api_key import APIKeyHeader, APIKey
from starlette.status import HTTP_403_FORBIDDEN
from config import settings
from models import QueryRequest
from dotenv import load_dotenv

from apiclient.discovery import build
from google.oauth2 import service_account

# Load environment variables from .env file
load_dotenv()

#config = yaml.safe_load(open("config.yml"))

#os.environ["OPENROUTER_API_KEY"] = config["OPENROUTER_API_KEY"]
#os.environ["LANGCHAIN_API_KEY"] = config["LANGCHAIN_API_KEY"]
#os.environ["LANGCHAIN_TRACING_V2"] = str(config["LANGCHAIN_TRACING_V2"]).lower()
#os.environ["LANGCHAIN_ENDPOINT"] = config["LANGCHAIN_ENDPOINT"]
#os.environ["LANGCHAIN_PROJECT"] = config["LANGCHAIN_PROJECT"]
#os.environ["LANGCHAIN_HUB_API_KEY"] = config["LANGCHAIN_API_KEY"]
#os.environ["LANGCHAIN_HUB_API_URL"] = config["LANGCHAIN_HUB_API_URL"]
#os.environ["GOOGLE_API_KEY"] = config["GOOGLE_API_KEY"]

#GOOGLE_API_KEY = config["GOOGLE_API_KEY"]

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]
    question: str
    intermediate_steps: str


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=4,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    # other params...
)


def waiting_msg(space_name):
    print(space_name)
    # Specify required scopes.
    SCOPES = ['https://www.googleapis.com/auth/chat.bot']

    # Specify service account details.
    CREDENTIALS = service_account.Credentials.from_service_account_file(
        'credentials.json', scopes=SCOPES)

    # Build the URI and authenticate with the service account.
    chat = build('chat', 'v1', credentials=CREDENTIALS)

    # Create a Chat message.
    result = chat.spaces().messages().create(

        # The space to create the message in.
        #
        # Replace SPACE with a space name.
        # Obtain the space name from the spaces resource of Chat API,
        # or from a space's URL.
        parent=space_name,

        # The message to create.
        body={'text': 'Estoy consultando mi base de conocimiento, esto puede demorar un rato ⏳...'}

    ).execute()

    print(result)


# Define the function that calls the model
def call_csv_agent(state):

    # AGENTE CSV
    path_csv = "https://docs.google.com/spreadsheets/d/1DJf-7PD5pje0YXSuYezBTglg_EFQb0C88WqJEFWVpe4/export?format=csv"

    csv_agent = create_csv_agent(
        llm,
        path_csv,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        allow_dangerous_code=True,
        return_intermediate_steps=True
    )

    question = state['messages'][0].content

    with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       ):
        response = csv_agent(question)

        output = response['output']

        global intermediate_steps_glob
        intermediate_steps_glob= response['intermediate_steps']

        print(response['intermediate_steps'])

        """for step in response['intermediate_steps'][0]:
            intermediate_steps = intermediate_steps + step.log"""
        intermediate_steps = response['intermediate_steps'][0][0].log

    # We return a list, because this will get added to the existing list
    return {"messages": [output], "question": question, "intermediate_steps": intermediate_steps}


template = """
Eres un agente amable y especializado en cambiar el formato de la información
que recibe de otro primer agente analítico.

La pregunta original que responde el primer agente es: {question}\n

A partir de la respuesta del agente analítico:

###########
Respuesta agente analítico:\n {input}\n
###########

Genere una respuesta manteniendo la información original y los nombres, pero cambie la 
estructura de tabla a una estructura de texto utilizando listas, sublistas, títulos, subtítulos, etc.

A partir de la siguiente información:

###########
Pasos realizados por el agente analítico: {intermediate_steps}\n
###########

Utilice únicamente la sección "Thought" par agregar al final de su respuesta una explicación de los pasos 
que realizó el agente analítico. Deberá indicar los pasos en un formato de lista de pasos.

Respuesta:\n
"""

# Define your custom prompt for the CSV agent
custom_prompt = PromptTemplate(
    input_variables=["input","question","intermediate_steps"],
    template=template
)



agent = (
    custom_prompt
    | llm
    | StrOutputParser()
)

def call_parse_output_agent(state):
    messages = state['messages']
    question = state['question']
    intermediate_steps = state['intermediate_steps']

    last_message = messages[-1]
    response_csv_agent = last_message.content

    response = agent.invoke({"input": response_csv_agent, "question": question, "intermediate_steps": intermediate_steps})

    return {"messages": [response]}


# Construimos el grafo
def build_graph():
    graph_builder = StateGraph(State)

    graph_builder.add_node("csv_agent", call_csv_agent)
    graph_builder.add_node("parse_output_agent", call_parse_output_agent)

    graph_builder.set_entry_point("csv_agent")

    graph_builder.add_edge('csv_agent', 'parse_output_agent')

    graph_builder.set_finish_point("parse_output_agent")

    graph = graph_builder.compile()

    return graph



graph = build_graph()

def use_bot_app(query):

    pregunta_1 = "Necesito una tabla con matrícula inicial total por línea formativa y por año. Las columnas deben ser los años y las filas las líneas formativas."
    pregunta_2 = "Necesito la cantidad total de cursantes en el año 2021"
    pregunta_3 = "Necesito una tabla con cantidad total de cursantes por tipo de carrera (filas) y por año (columnas)."
    pregunta_4 = "Necesito una tabla matricula inicial y cursantes agrupados por IFDA y por los años 2022 y 2023."
    pregunta_5 = "Necesito una tabla con total de inscriptos para la carrera con siglas TRAYECTO_PED, cohorte abril 2024, agrupados por IFDA y siglas del IFDA."

    inputs = {"messages": [HumanMessage(content=query)]}

    ans = graph.invoke(inputs)

    return ans


# Configure logging
#logging.basicConfig(level=logging.INFO)
#logger = logging.getLogger(__name__)

API_KEY_NAME = "Authorization"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

app = FastAPI()
print(api_key_header)

async def get_api_key(api_key_header: str = Depends(api_key_header)):
    if api_key_header and api_key_header.split(" ")[1] == settings.api_key:
        print("COSO")
        return api_key_header
    else:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="Could not validate credentials"
        )


@app.get("/public")
def read_public():
    return {"message": "This is a public endpoint"}


@app.post("/protected")
def read_protected(
    text: QueryRequest,
    api_key: APIKey = Depends(get_api_key)
):
    #logger.info(f"Received text: {text.text}")
    
    waiting_msg(text.space_name)
    ans = use_bot_app(text.text)
    #print(ans["messages"][-1].content)
    clean_msg = str(ans["messages"][-1].content).replace('###','*').replace('##','*').replace('##','*').replace('**','*')

    return {"message": "respuesta del bot", "text": clean_msg}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)