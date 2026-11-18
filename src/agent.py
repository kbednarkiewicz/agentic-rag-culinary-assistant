from dotenv import load_dotenv
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
from src.config import LLM_MODEL_NAME
from src.tools import search_recipes, ingredient_converter

load_dotenv()

SYSTEM_PROMPT="""
You are a helpful culinary assistant on a desert island. Survivors of the Titanic disaster need to be fed so always refer to people as "survivors".
Answer questions strictly based on the provided PDF recipe base and available tools.
Always use 'search_recipes' for questions about recipes or ingredients. When the user asks for a recipe, provide the complete recipe found in the PDF.
If the recipe is incomplete do not include it in your answer.
If information is missing from the documents, clearly state that it was not found.
Use 'ingredient_converter' whenever ingredient amounts need to be adjusted for a different number of servings.
"""
def parse_result(res):
	to_parse = res["messages"][-1]
	if isinstance(to_parse.content, list):
		return to_parse.content[0]["text"]
	return str(to_parse.content)

class CulinaryAgent:
	def __init__(self):
		self.tools = [search_recipes, ingredient_converter]
		self.agent = create_agent(model=LLM_MODEL_NAME, tools=self.tools, system_prompt=SYSTEM_PROMPT, checkpointer=InMemorySaver())

	def run_agent(self, query: str, thread_id: str="1"):
		config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
		result = self.agent.invoke({"messages": [{"role": "user", "content": query}]}, config)
		return parse_result(result)