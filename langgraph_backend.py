from langgraph.graph import StateGraph,START,END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage,AIMessage, BaseMessage
from typing import TypedDict,Literal,Annotated
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()
llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState) -> ChatState:
    response = llm.invoke(state['messages'])
    return {"messages": [response]}

checkpointer=MemorySaver()
graph = StateGraph(ChatState)
graph.add_node("chat_node",chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)
chatbot = graph.compile(checkpointer=checkpointer)