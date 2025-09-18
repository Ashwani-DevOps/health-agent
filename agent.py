from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType

# Use community version of Ollama if needed
try:
    from langchain.llms import Ollama
except ImportError:
    from langchain_community.llms import Ollama

from langchain.tools import DuckDuckGoSearchRun

# Initialize local LLM (make sure Ollama is installed and running)
llm = Ollama(model="mistral")  # You can use "llama2", "gemma", etc.

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
    verbose=True,
    handle_parsing_errors=True  # Add this line
)
#print("Ashwani:")
# Run agent
# Run agent
try:
    query = "What is the BMI for someone who weighs 70kg and is 1.75m tall?"
    response = agent.invoke(query)  # Updated method
    print("\nAgent Response:\n", response)
except Exception as e:
    print("Error:", e)
