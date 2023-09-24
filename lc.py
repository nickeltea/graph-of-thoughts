import os

import openai
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.agents import (AgentExecutor, AgentType, Tool, create_csv_agent,
                              initialize_agent, tool)
from langchain.llms import OpenAI
from langchain.llms.openai import OpenAI
from langchain.tools import StructuredTool

from chroma import *
from webscraper import *
from source_txt import *
from github_issues import *
from util import *

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def start_langchain():
    # Prompt
    print("Starting LangChain...")

    # QA Prompt
    csv_template = """As a malware identifying support bot, your goal is to provide 
        accurate and helpful binary analysis information by analysing the binary data 
        provided. You should answer user inquiries based on the context provided and avoid 
        making up answers. If you don't know the answer, simply state that you don't know. 
        Remember to provide relevant information about binary analysis to assist the user 
        in understanding the binary data. Try to consider all data provided.

        {context}

        Question: {question}"""

    CSV_PROMPT = PromptTemplate(
        template=csv_template, input_variables=["context", "question"]
    )
   
    llm = OpenAI(
        temperature=0,
        max_tokens=512,
        batch_size=5
    )

    # Get Github issues
    issues_filename = CSV_issues()

    csv_agent_executor = create_csv_agent(
        llm, 
        issues_filename, 
        verbose=True,
        chain_type_kwargs={"prompt": CSV_PROMPT}
    )

    @tool
    def CSV_agent_tool(query):
        """useful for when a user is interested in information about IP addresses, alerts or time.  
        If information about ports is needed, checks both destination and source ports 
        unless specified otherwise."""
        try:
            return csv_agent_executor.run(query)
        except openai.InvalidRequestError as e:
            return temp_chunks_CSV_agent(query, data = issues_filename, llm = llm)

    # The zero-shot-react-description agent will use the "description" string to select tools
    tools = [
        Tool(
            name = "CSV_agent",
            func=CSV_agent_tool,
            description="""useful for when a user is interested in problems with the program. 
            Input should be a fully formed question."""),
        # Tool(
        #     name = "source_search",
        #     func=source_questions,
        #     description="""Useful for when a user is interested in source code analysis. 
        #     Input should be a fully formed question."""),
        # Tool(
        #     name = "web_search",
        #     func=db_QA,
        #     description="""Useful for when a user wants to know about the WAV file format. 
        #     Input should be a fully formed question."""),
        Tool(
            name = "combined_search",
            func=combined_search,
            description="""Useful for when a user is interested in source code analysis or the WAV file format. 
            Input should be a fully formed question."""),
        Tool(
            name = "error_handling",
            func=error_handling,
            description="""Useful for when an error occurs and the user must be informed."""),
    ]

    # Zero-Shot Agent
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)


    # Test
    inpt = input("What would you like to know about the binary?\nType 'exit' to exit.\n")
    while inpt.strip() != "exit":
        
        try:
            agent.run(inpt)
            inpt = input("What would you like to know about the binary?\nType 'exit' to exit.\n")
        except openai.InvalidRequestError as e:
            print(e)
            inpt = input("What would you like to know about the binary?\nType 'exit' to exit.\n")

    os.unlink(issues_filename)

start_langchain()
