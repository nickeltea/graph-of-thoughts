import os
import pprint

import openai
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import (AsyncChromiumLoader, AsyncHtmlLoader,
                                        TextLoader)
from langchain.document_transformers import (BeautifulSoupTransformer,
                                             Html2TextTransformer)
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms.openai import OpenAI
from langchain.text_splitter import (CharacterTextSplitter,
                                     RecursiveCharacterTextSplitter)
from langchain.vectorstores import Chroma

load_dotenv()

# Setup
openai.api_key = os.getenv("OPENAI_API_KEY")
urls = ['https://en.wikipedia.org/wiki/WAV']

loader = AsyncHtmlLoader(urls)
docs = loader.load()

html2text = Html2TextTransformer()
docs_transformed = html2text.transform_documents(docs)

text_splitter = CharacterTextSplitter(        
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
    )

texts = text_splitter.split_documents(docs_transformed)
embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings)

llm = OpenAI(
    temperature=0,
    max_tokens=1024,
    batch_size=60
)

# Prompt for WAV info search
search_template = """Answer the question below. You are provided with information about the WAV file format
to help.

{context}

Question: {question}"""

SEARCH_PROMPT = PromptTemplate(template=search_template, input_variables=["context", "question"])

search_qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        chain_type_kwargs={"prompt": SEARCH_PROMPT},
)

def db_QA(query):
    return search_qa.run(query)