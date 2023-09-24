import csv
import json
import logging
import os
import subprocess

import openai
import pandas as pd
import pyshark
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from scapy.all import *
from scapy.all import TCP

from langchain import PromptTemplate
from langchain.agents import create_csv_agent, create_pandas_dataframe_agent
from langchain.chains import RetrievalQA
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import (
                                     RecursiveCharacterTextSplitter)
from langchain.vectorstores import Chroma

logging.getLogger("scapy.runtime").setLevel(logging.ERROR)
format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
def prepare_logs_folder(logs_folder):
    # Create the logs folder if it doesn't exist
    if not os.path.exists(logs_folder):
        print(f"{logs_folder} does not exist. Creating logs folder")
        os.makedirs(logs_folder)

    # Delete all files in the logs folder
    file_list = os.listdir(logs_folder)
    print("Clearing logs folder")
    for file_name in file_list:
        file_path = os.path.join(logs_folder, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)

def create_output_directory_folder(directory_name, output_directory='objects') -> str:
    if not os.path.exists(output_directory):
        logging.debug('Directory %s does not exists - creating' % output_directory)
        os.mkdir(output_directory)
    directory_name =  directory_name.replace('.pcap', '')
    target_path = os.path.join(os.getcwd(),output_directory, directory_name)
    if not os.path.exists(target_path):
        logging.debug('Path %s does not exists - creating.' % target_path)
        os.mkdir(target_path)
    return target_path

def get_http_headers(http_payload):
    try:
        headers_raw = http_payload[:http_payload.index(b"\r\n\r\n") + 2]
        headers = dict(re.findall(b"(?P<name>.*?): (?P<value>.*?)\\r\\n", headers_raw))

    except ValueError as err:
        logging.error('Could not find \\r\\n\\r\\n - %s' % err)
        return None
    
    except Exception as err:
        logging.error('Exception found trying to parse raw headers - %s' % err)
        logging.debug(str(http_payload))
        return None

    if b"Content-Type" not in headers:
        logging.debug('Content Type not present in headers')
        logging.debug(headers.keys())
        return None   
    return headers

def convert_json_to_csv(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Create an empty DataFrame
    df = pd.DataFrame()

    for key, value in data.items():
        packet_number = int(key.split(' ')[1])
        flattened_data = pd.json_normalize(value, sep='_')
        flattened_data.insert(0, 'Packet Number', packet_number)
        df = df._append(flattened_data, ignore_index=True)

    # Write DataFrame to CSV
    # Define the output CSV file path
    csv_filepath = f'{json_file[:-5]}_csv.csv'

    try:
        # Write DataFrame to CSV
        df.to_csv(csv_filepath, index=False)
        if os.path.isfile(csv_filepath):  # Check if the file exists
            return csv_filepath
    except Exception as e:
        print(f"Error occurred while converting JSON to CSV: {e}")
        return False
    
def scrape_text_into_csv(url, output_file="scraped_output.csv"):
    # Send a GET request to the website
    response = requests.get(url)

    # Create BeautifulSoup object to parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find specific elements containing Trickbot-related information
    article_content = soup.find('article')

    # Filter and extract relevant information
    relevant_information = []

    # Example: Extract information within <p> tags
    paragraphs = article_content.find_all('p')
    for p in paragraphs:
        # Filter based on specific patterns or keywords
        if p.get_text().lower().strip().startswith('figure'):\
            continue

        # if 'trickbot' in p.get_text().lower():
        #     relevant_information.append(p.get_text())
        relevant_information.append(p.get_text())

    # Extract information within <ul> tags
    unordered_lists = article_content.find_all('ul')
    for ul in unordered_lists:
        list_items = ul.find_all('li')
        for li in list_items:
            relevant_information.append(li.get_text())

    # Extract information within <ol> tags
    ordered_lists = article_content.find_all('ol')
    for ol in ordered_lists:
        list_items = ol.find_all('li')
        for li in list_items:
            relevant_information.append(li.get_text())

     # Open the output file in write mode with CSV writer
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)

        # Write the relevant information to the CSV file
        for info in relevant_information:
            writer.writerow([info])

    print(f"Scraped data has been saved to '{output_file}'.")

# scrape_text_into_csv("https://unit42.paloaltonetworks.com/wireshark-tutorial-examining-trickbot-infections/","trickbot_scraped.csv")
def get_infection_info(query):
    
    persist_directory = 'db'
    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
    
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    llm = OpenAI(temperature=0.2, openai_api_key=OPENAI_API_KEY)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectordb.as_retriever(search_kwargs={"k": 5}))

    return (qa.run(query))

def upload_scraped_data(url, outputfile="scraped_output.csv"):
    scrape_text_into_csv(url, output_file=outputfile)


    loader = CSVLoader(outputfile)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap  = 200,
        length_function = len,
    )

    docs = text_splitter.split_documents(data)

    # Embed and store the texts
    # Supplying a persist_directory will store the embeddings on disk
    print("Initialising Chroma DB")
    persist_directory = 'db'
    embedding = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory=persist_directory)
    vectordb.persist()
    vectordb = None

def error_handling(response):
    return response

def temp_chunks_CSV_agent(query, data, llm):
    # Prompt
    csv_template = """
    Your answer should only be Yes or No.

    Answer: {answer}"""

    CSV_PROMPT = PromptTemplate(
        template=csv_template, input_variables=["answer"]
    )

    for chunk in pd.read_csv(data, chunksize=10):

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, dir=os.getcwd()) as temp_file:
            # Write the DataFrame to the temporary file
            chunk.to_csv(temp_file.name, index=False)
        
            # Initialise CSV Agent for each batch
            agent = create_csv_agent(
            llm, 
            temp_file.name, 
            verbose=True,
            chain_type_kwargs={"prompt": CSV_PROMPT}
            )

            result = agent.run(query)

            temp_file.close()
            os.unlink(temp_file.name)

            if 'yes' in result.lower():
                return "Yes"
    
    return "No"
