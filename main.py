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

# Initialize session state for debate
if 'debate_state' not in st.session_state:
    st.session_state.debate_state = None
if 'current_turn' not in st.session_state:
    st.session_state.current_turn = 0
if 'debate_history' not in st.session_state:
    st.session_state.debate_history = []
if 'all_turns' not in st.session_state:
    st.session_state.all_turns = []

def start_new_debate():
    # Initialize debate state
    initial_state = {"topic": topic, "history": [], "turn": 0, "max_turns": max_turns}
    
    # Run the entire debate once
    final_state = app.invoke(initial_state)
    
    # Store all turns
    st.session_state.all_turns = final_state["history"]
    st.session_state.current_turn = 1
    st.session_state.debate_history = [st.session_state.all_turns[0]]

def next_turn():
    if st.session_state.current_turn < len(st.session_state.all_turns):
        st.session_state.current_turn += 1
        st.session_state.debate_history = st.session_state.all_turns[:st.session_state.current_turn]

if st.button("Start Debate"):
    start_new_debate()
    st.rerun()

# Display debate progress
if st.session_state.all_turns:
    st.subheader("🎤 Debate Progress")
    st.progress(st.session_state.current_turn / max_turns)
    st.write(f"Turn {st.session_state.current_turn} of {max_turns}")

# Display current turn only
if st.session_state.debate_history:
    st.subheader("🎤 Current Turn")
    current_message = st.session_state.debate_history[-1]
    label = "🟢 FOR" if len(st.session_state.debate_history) % 2 != 0 else "🔴 AGAINST"
    st.markdown(f"**{label}:** {current_message.content}")

# Show Next Turn button only if there are more turns
if (st.session_state.all_turns and 
    st.session_state.current_turn < len(st.session_state.all_turns)):
    if st.button("Next Turn"):
        next_turn()
        st.rerun()

# Show completion message
if st.session_state.all_turns and st.session_state.current_turn >= len(st.session_state.all_turns):
    st.success("🎉 Debate completed! Click 'Start Debate' to begin a new one.")








