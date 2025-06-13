from langchain_groq import ChatGroq
from typing import TypedDict, List, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import streamlit as st
import os

load_dotenv()

llm = ChatGroq(model_name="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"))

FOR_PROMPT = "You are a persuasive debater who always argues in favor of the topic."
AGAINST_PROMPT = "You are a persuasive debater who always argues against the topic."

class DebateState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    history: Annotated[List[BaseMessage], add_messages]
    topic: str
    max_turns: int
    turn: int
    last_speaker: str

def for_agent(state: DebateState) -> DebateState:
    history = state.get("history", [])
    topic = state["topic"]
    messages = history + [SystemMessage(content=FOR_PROMPT), HumanMessage(content=f"The topic is: {topic}")]
    response = llm(messages)
    history.append(HumanMessage(content=response.content))
    return {"history": history, "last_speaker": "for", "turn": state["turn"] + 1, "topic": topic, "max_turns": state["max_turns"]}

def against_agent(state: DebateState) -> DebateState:
    history = state.get("history", [])
    topic = state["topic"]
    messages = history + [SystemMessage(content=AGAINST_PROMPT), HumanMessage(content=f"The topic is: {topic}")]
    response = llm(messages)
    history.append(HumanMessage(content=response.content))
    return {"history": history, "last_speaker": "against", "turn": state["turn"] + 1, "topic": topic, "max_turns": state["max_turns"]}

def route(state: DebateState) -> str:
    if state["turn"] >= state["max_turns"]:
        return END
    else:
        return "for_agent" if state["last_speaker"] == "against" else "against_agent"

graph = StateGraph(DebateState)
graph.add_node("for_agent", for_agent)
graph.add_node("against_agent", against_agent)
graph.set_entry_point("for_agent")
graph.add_conditional_edges("for_agent", route)
graph.add_conditional_edges("against_agent", route)

app = graph.compile()

#Streamlit UI
st.title("🧠 AI Debate App")
topic = st.text_input("Enter a topic for the debate:", value="Should AI replace teachers?")
max_turns = st.slider("Number of Debate Turns (2-10)", min_value=2, max_value=10, value=6, step=2)

if st.button("Start Debate"):
    with st.spinner("Debating..."):
        initial_state = {"topic": topic, "history": [], "turn": 0, "max_turns": max_turns}
        final_state = app.invoke(initial_state)

        st.subheader("🎤 Debate Transcript")
        history = final_state["history"]
        for i, msg in enumerate(history, 1):
            label = "🟢 FOR" if i % 2 != 0 else "🔴 AGAINST"
            st.markdown(f"**Turn {i} ({label}):** {msg.content}")








