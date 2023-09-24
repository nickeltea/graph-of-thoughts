import os

import openai
from db_source_txt import *
from db_webscraper import *
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings()
database = Chroma(collection_name="source_webscraped_db", embedding_function=embeddings)

# update db
source_data = pass_source_info()
scraped_data = pass_WAV_info()
database.add_documents(source_data)
database.add_documents(scraped_data)

llm = OpenAI(
    temperature=0,
    max_tokens=1024,
    batch_size=60
)

# Prompt for WAV info search
search_template = """Answer the question below. You are provided with information about the 
WAV file format and source code to help.

{context}

Question: {question}"""

SEARCH_PROMPT = PromptTemplate(template=search_template, input_variables=["context", "question"])

search_qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=database.as_retriever(),
        chain_type_kwargs={"prompt": SEARCH_PROMPT},
)

def combined_search(query):
    return search_qa.run(query)

def source_based_questions(query):
    return source_questions(query)

def web_based_questions(query):
    return webscraper(query)

qn = "what is WAV?"
print(combined_search(qn))