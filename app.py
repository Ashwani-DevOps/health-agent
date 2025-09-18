import streamlit as st
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType

# Use community version of Ollama if needed
try:
    from langchain.llms import Ollama
except ImportError:
    from langchain_community.llms import Ollama

from langchain.tools import DuckDuckGoSearchRun

# Initialize LLM
llm = Ollama(model="mistral")

# Define tools
search = DuckDuckGoSearchRun()
calculator = Tool(
    name="Calculator",
    func=lambda x: str(eval(x)),
    description="Useful for math calculations"
)

# Create agent
agent = initialize_agent(
    tools=[search, calculator],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

# Streamlit UI
st.title("🩺 Local Health Agent")
query = st.text_input("Ask a health-related question:")

if query:
    with st.spinner("Thinking..."):
        try:
            response = agent.invoke(query)
            st.success(response)
        except Exception as e:
            st.error(f"Error: {e}")