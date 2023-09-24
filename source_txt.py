import os

import openai
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.llms import OpenAI

from util import *

load_dotenv()

data_source = "wavdec copy.txt"

openai.api_key = os.getenv("OPENAI_API_KEY")

raw_documents = TextLoader(data_source).load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)

# Define embedding model
data_store = Chroma.from_documents(documents, OpenAIEmbeddings())

llm = OpenAI(
    temperature=0,
    max_tokens=1024,
    batch_size=60
)

# Prompt for returning threat characteristics as questions
source_template = """Answer the question below. You are provided source code 
to help with answering this question.

{context}

Question: {question}"""

SOURCE_PROMPT = PromptTemplate(template=source_template, input_variables=["context", "question"])

binary_qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=data_store.as_retriever(),
        chain_type_kwargs={"prompt": SOURCE_PROMPT},
)

def source_questions(query):
    return binary_qa.run(query)

qn = "from the source code, describe the wav file format?"
print(source_questions(qn))