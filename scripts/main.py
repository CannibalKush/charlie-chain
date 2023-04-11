# Do anything bot using LangChain.

# Set up environment

from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser, load_tools
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
from langchain.document_loaders import DirectoryLoader
loader = DirectoryLoader('./files/', glob="*")
from langchain.utilities import BashProcess

import re

import os
os.environ["OPENAI_API_KEY"] = ""
os.environ["SERPAPI_API_KEY"] = ""

# Set up the base template
template = """You are a fully automated Large Language Model Chain. You are tasked to reason about a question, analyse it, then use one of the following tools to move towards solving it:

{tools}

Your response must be in the following format:

Question: The input question you must answer
Thought: Any pertinent thoughts you have about the question
Criticism: Any criticism of the action to be taken -- name risks or concerns
Action: An action to take, which must be one of [{tool_names}]
Action Input: The input to the action
Observation: The result of the action

This response will in turn be used as the input to the next agent in the chain.
When you feel as though the question has been answered, or that you're ready to give up, you may give your final answer in the following format:

Thought: I now know the final answer
Conclusion: A short description of actions taken or an answer to a question

The following question is the first in the chain. You may begin. Ensure that you confirm the accuracy of your answer to the best of your abilities. Do not simply rely on the tools you have available, and use your cognition to the best of your abilities. Some common sense is required.
Feel free to sometimes omit very obvious steps, but do not omit steps that are not obvious.

Question: {input}
{agent_scratchpad}"""

# Set up a prompt template


class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)


class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Conclusion:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split(
                    "Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)


output_parser = CustomOutputParser()

llm = OpenAI(temperature=0.65)

# Define which tools the agent can use to answer user queries
search = SerpAPIWrapper()
bash = BashProcess()


tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Search for information on current events and factual information. Does not work well for questions that require reasoning.",
    ),
    Tool(
        name="Bash",
        func=bash.run,
        description="Run a bash command on the local machine. It gives you access to the entire file system."
    )
]


prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps"]
)

# LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(llm=llm, prompt=prompt)

tool_names = [tool.name for tool in tools]

agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True)

# agent_executor.run(
#     "What is the output of `ls`?")
print(bash.run("dir"))
