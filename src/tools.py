from pathlib import Path
from langchain_core.tools import tool
from src.ingest import load_vectorstore
from src.config import RETRIEVER_K
_retriever = None

def _get_retriever():
    global _retriever
    if _retriever is None:
        vectorstore = load_vectorstore(force_rebuild=False)
        _retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
    return _retriever

@tool
def search_recipes(query: str) -> str:
    """
    Search the PDF knowledge base for recipe details.
    Use this for any questions about recipes, ingredients, cooking instructions, or preparation time.
    Answer based strictly only on the PDF documents. Always provide the source of the information.
    """
    print("DEBUG: search_recipes used")
    retriever = _get_retriever()
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information was found in the recipe knowledge base."

    results = []

    for doc in docs:
        source = Path(doc.metadata["source"]).name
        page = doc.metadata.get("page", 0) + 1

        results.append(
            f"Source: {source}, page: {page}\n"
            f"{doc.page_content}"
        )

    return "\n\n---\n\n".join(results)

@tool
def ingredient_converter(
    quantity: float,
    original_servings: int,
    new_servings: int) -> float:
    """
    Calculate a new ingredient quantity when changing the number of servings.
    Use this tool whenever the user asks to scale an ingredient.
    Use this tool for all quantities, including item counts.
    You can only do the calculations.
    """

    print("DEBUG: ingredient_converter used")

    if original_servings <= 0 or new_servings <= 0:
        raise ValueError("Number of servings must be greater than zero.")
    if quantity < 0:
        raise ValueError("Quantity cannot be negative.")
    return quantity / original_servings * new_servings
