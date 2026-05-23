from langgraph.graph import StateGraph,START,END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage,AIMessage, BaseMessage
from typing import TypedDict,Literal,Annotated
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver,sqlite3

load_dotenv()
llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState) -> ChatState:
    response = llm.invoke(state['messages'])
    return {"messages": [response]}

conn=sqlite3.connect(database="chatbot.db",check_same_thread=False)
checkpointer=SqliteSaver(conn=conn)

CONFIG = {'configurable': {'thread_id': 'thread-1'}}
graph = StateGraph(ChatState)
graph.add_node("chat_node",chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)
chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads=set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return all_threads
