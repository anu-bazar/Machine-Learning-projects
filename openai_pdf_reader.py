"""Using chatgpt from the console"""

import os
import openai
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma

#import the token
#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
os.environ["OPENAI_API_KEY"] = "INSERT KEY HERE"
openai.api_key = os.environ['OPENAI_API_KEY']

#https://pubmed.ncbi.nlm.nih.gov/36940333/
# Using a paper
paper = PyPDFLoader('dmt.pdf')
paper = paper.load()
#split data into chunks, not using up all the tokens
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
banana = splitter.split_documents(paper)
#define model and temperature
bedding = OpenAIEmbeddings()
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
#temp - higher: somehow randomize the answer, 0: it will give the same answer

assert isinstance(Chroma.from_documents, object)
db = Chroma.from_documents(documents=banana, embedding= bedding, persist_directory="chroma")
#define template
question = "What was the goal of the experiment?"
template = """Answer the question below, based on the context. Use five sentences maximum.
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

#set up the QA chain

qa_chain = RetrievalQA.from_chain_type(
    lim,
    retriever = db.as_retriever(),
    chain_type_kwargs={"prompt":QA_CHAIN_PROMPT}
)
act= qa_chain({"query":question})
res=act["result"]
print(res)


"""Homework: analyze an article based on this today"""





