from langchain.tools import Tool
from typing import List, Generator
import re

def format_observation(observation : str) :

    return "Observation :" + observation


def format_tools(tools : List[Tool]):

    return "\n\n".join((f"{tool.name} : {tool.description}" for tool in tools))

_tool_name_extraction_re = re.compile("[tT]ool\s*:?\s*([^\n\r]*)")
_tool_input_extraction_re = re.compile("[tT]ool [iI]nput\s*:?\s*([^\n\r]*)")
# _name_capture_re = re.compile("([\w\d_])\((.*)\)")

def parse_output(output : str, tool_names : List[str],tools : List[Tool]):

    if output.find("Final Answer:") != -1:
        return "end", None
    
    print("output :",output)

    tool_extraction_match = _tool_name_extraction_re.findall(output)

    if len(tool_extraction_match) >= 2:
        action_name = tool_extraction_match[-2]
    else:
        raise ValueError("The LLM did not follow the prompt correctly and no tool was selected :( . Here is what the LLM returned : \n" + output)

    action_name = action_name.replace("\\", "")
    tool_input = _tool_input_extraction_re.findall(output)[-1]

    try:
        tool_index = tool_names.index(action_name)
        selected_tool = tools[tool_index]

        return selected_tool, tool_input    
    except ValueError:
        print("Invalid selected tool :", action_name)
        raise ValueError("The LLM selected an invalid tool :( . Here is what the LLM returned : \n" + output)
    # if len(selected_tool.args_schema.dict()) == 1:

    #     tool_input = splitting[1]
    # elif len(selected_tool.args_schema.dict()) == 0:
    #     tool_input = None

    # else:
    #     raise ValueError("Cock")


def run_tool(selected_tool, tool_input):
    if tool_input is None:
        return selected_tool.run()
    return selected_tool.run(tool_input)

def display_return_response(response_stream : Generator[str,None,None]) -> str:
    acc = ""
    for token in response_stream:
        print(token, end="")
        acc += token
    return acc

def execute_agent(llm, question : str, tools : List[Tool], prompt_template : str):

    print("Starting agent execution...\n Thought: ")

    tool_names = [tool.name for tool in tools]

    prompt = prompt_template.format(
        tools = format_tools(tools),
        query = question,
        transcript = "I should select a tool to use using the provided template. "
    )


    while True:

        next_step : str = display_return_response(llm(prompt, stop = ["Observation:"], stream = True, temperature = 0))

        selected_tool, tool_input = parse_output(next_step, tool_names, tools)

        if selected_tool == "end":
            print("Finished agent execution")
            return

        observation = run_tool(selected_tool, tool_input)

        prompt +=  next_step.rstrip() + "\n" + format_observation(observation) + "\nThought: "




FEW_SHOT_REACT_TEMPLATE = """
You are a helpful and knowledgeable agent. To achieve your goal of answering complex questions correctly, you have access to the following tools:
Search: useful for when you need to Google questions. You should ask targeted questions, for example, Who is Anthony Dirrell's brother?
To answer questions, you'll need to go through multiple steps involving step-by-step thinking and selecting appropriate tools and their inputs; tools will respond with observations. When you are ready for a final answer, respond with the `Final Answer:`
Examples:
##
Question: Anthony Dirrell is the brother of which super middleweight title holder?
Thought: Let's think step by step. To answer this question, we first need to know who Anthony Dirrell is.
Tool: Search
Tool Input: Who is Anthony Dirrell?
Observation: Boxer
Thought: We've learned Anthony Dirrell is a Boxer. Now, we need to find out who his brother is.
Tool: Search
Tool Input: Who is Anthony Dirrell brother?
Observation: Andre Dirrell
Thought: We've learned Andre Dirrell is Anthony Dirrell's brother. Now, we need to find out what title Andre Dirrell holds.
Tool: Search
Tool Input: What is the Andre Dirrell title?
Observation: super middleweight
Thought: We've learned Andre Dirrell title is super middleweight. Now, we can answer the question.
Final Answer: Andre Dirrell
##
Question: What year was the party of the winner of the 1971 San Francisco mayoral election founded?
Thought: Let's think step by step. To answer this question, we first need to know who won the 1971 San Francisco mayoral election.
Tool: Search
Tool Input: Who won the 1971 San Francisco mayoral election?
Observation: Joseph Alioto
Thought: We've learned Joseph Alioto won the 1971 San Francisco mayoral election. Now, we need to find out what party he belongs to.
Tool: Search
Tool Input: What party does Joseph Alioto belong to?
Observation: Democratic Party
Thought: We've learned Democratic Party is the party of Joseph Alioto. Now, we need to find out when the Democratic Party was founded.
Tool: Search
Tool Input: When was the Democratic Party founded?
Observation: 1828
Thought: We've learned the Democratic Party was founded in 1828. Now, we can answer the question.
Final Answer: 1828
##
Now you have to following tools:
{tools}
Question: {query}
Thought: {transcript}
"""

FEW_SHOT_REACT_TEMPLATE_SMALL = """
You are a helpful and knowledgeable agent. To achieve your goal of answering complex questions correctly, you have access to the following tools:
Search: useful for when you need to Google questions. You should ask targeted questions, for example, Who is Anthony Dirrell's brother?
To answer questions, you'll need to go through multiple steps involving step-by-step thinking and selecting appropriate tools and their inputs; tools will respond with observations. When you are ready for a final answer, respond with the `Final Answer:`
Example:
##
Question: Anthony Dirrell is the brother of which super middleweight title holder?
Thought: Let's think step by step. To answer this question, we first need to know who Anthony Dirrell is.
Tool: Search
Tool Input: Who is Anthony Dirrell?
Observation: Boxer
Thought: We've learned Anthony Dirrell is a Boxer. Now, we need to find out who his brother is.
Tool: Search
Tool Input: Who is Anthony Dirrell brother?
Observation: Andre Dirrell
Thought: We've learned Andre Dirrell is Anthony Dirrell's brother. Now, we need to find out what title Andre Dirrell holds.
Tool: Search
Tool Input: What is the Andre Dirrell title?
Observation: super middleweight
Thought: We've learned Andre Dirrell title is super middleweight. Now, we can answer the question.
Final Answer: Andre Dirrell
Now you have to following tools:
{tools}
Question: {query}
Thought: {transcript}
"""


if __name__ == "__main__":

    from langchain.tools.file_management import ReadFileTool
    from typing import Any
    from search_tool import SearchTool
    from llm_client import LocalEmbeddingsClient, LocalLLMClient
    from pydantic import BaseModel

    class TellJokeTool(BaseModel):

        name : str = "tell_joke_tool"
        description : str = "A tool that allows you to tell a funny joke about a given topic. The input of this tool is the topic you want to hear a joke about"
        llm : Any

        template : str = (
            "You are a funny comedian that is famous for its contreversial and dirty jokes. "
            "You are not afraid of using curse words like 'fuck' and 'shit'. "
            "Here is a joke topic :{topic}.\n"
            "Write a funny joke about it. Go!"
        )

        def run(self, topic : str) -> str:

            prompt = self.template.format(topic = topic)

            return self.llm(prompt)

        async def run_async(self, topic : str) -> str:

            prompt = self.template.format(topic=topic)

            return await self.llm._call_async(prompt)
        
    class TellStory(BaseModel):

        name : str = "tell_story_tool"
        description : str = "A tool that allows you to tell a short story about a given text. The input of this tool is the text you want to hear a story about"
        llm : Any

        template : str = (
            "You are a skilled story teller that is famous for its touching stories. "
            "Here is a price of text :{text}.\n"
            "Write a short story in relation to the text. Please adapt the tone of your story in function of the tone of the text Go!"
        )

        def run(self,text : str) -> str:

            prompt = self.template.format(topic = text)

            return self.llm(prompt)

        async def run_async(self, text : str) -> str:

            prompt = self.template.format(text=text)

            return await self.llm._call_async(prompt)
    
    encoder = LocalEmbeddingsClient("http://127.0.0.1:8000")
    
    # we are using mistral instruct 7b prompt template
    llm = LocalLLMClient("http://127.0.0.1:8000", prompt_template = "<s>[INST] {prompt} [/INST]")


    tools = [ReadFileTool(), TellJokeTool(llm = llm), TellStory(llm=llm), SearchTool(llm=llm, encoder=encoder)]


    execute_agent(llm, "Please tell me a story about the Gaza's latest news", tools, FEW_SHOT_REACT_TEMPLATE_SMALL)