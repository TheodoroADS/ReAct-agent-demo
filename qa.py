from llm_client import LocalLLMClient
from ctransformers import AutoModelForCausalLM
from typing import Sequence
from VectorDb import VectorDB

# The type for a local llm 
LLM = LocalLLMClient | AutoModelForCausalLM

DEFAULT_QA_SYSTEM_PROMPT_TEMPLATE = (
'''
You are a a knowledgable agent working with data from the web. An user made a web search with a given query and got 
a set of documents as a response. Your task is to answer the user's query using the search results.

- Be as concise as possible.
- Be as precise as possible.
- Use only the information contained inside the search results to answer the query.
- If you cannot answer using the search results, simply say "No relevant information was found on the web."

Here is the user query : {query}

Here are the search results:

BEGIN SEARCH RESULTS
{sources}
END SEARCH RESULTS

Answer the query. Go!

''')

def format_search_results(search_results : Sequence[str]) -> str:

    return "\n".join(search_results)

def answer_query(
        query : str, 
        llm : LLM, 
        index : VectorDB, 
        prompt_template : str = DEFAULT_QA_SYSTEM_PROMPT_TEMPLATE,
        nb_documents : int = 2):
    
    search_results = index.similarity_search(query, k = nb_documents)

    print("Generating response from sources")
    prompt = prompt_template.format(
        query = query,
        sources = format_search_results(search_results)
    )

    model_response = llm(prompt, temperature = 0)

    return model_response

