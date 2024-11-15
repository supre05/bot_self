'''
Python Script for the implementation of the AI agent. RAG, Web Search, Direct.
'''

# Import libraries
import os
from threading import Thread, Lock
from functools import lru_cache
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel,Field
from typing import Literal
from langchain_community.cache import InMemoryCache
from langchain.globals import set_llm_cache
from langchain_community.tools.tavily_search import TavilySearchResults

# Env 
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# Setup
embeddings = OpenAIEmbeddings(model = "text-embedding-3-large" , api_key = openai_api_key)
llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=1000,
            timeout=None,
            max_retries=2,
            api_key=openai_api_key,
            # other params if needed..
        )


# Set Cache for speed
set_llm_cache(InMemoryCache())


def _index_docs(rag_files):
    # Load, chunk, and index the contents of the data
    docs = []
    for file in rag_files:
        data_file_path = file
        loader = None
        if file.endswith(".pdf"): # Add pdfs
            loader = PyPDFLoader(data_file_path)
        elif file.endswith(".txt"): # Add text files
            loader = TextLoader(data_file_path)
        docs.extend(loader.load())

    # Split and chunk the text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200, add_start_index = True)
    splits = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(documents = splits, embedding = embeddings) # Store in FAISS

    return vectorstore

def setup_vectorstore():
    # Setup vectorstore for RAG
    # memory = MemorySaver(
    # )    
    rag_files = ["./Data/Database_files/sample.pdf"]
    vectorstore_path = "./Data/vectorstore"
    try:
        vectorstore = FAISS.load_local(vectorstore_path, embeddings = embeddings, allow_dangerous_deserialization = True)
    except:
        # print('Creating a new vectorstore')
        vectorstore = _index_docs(rag_files) # Index docs
        vectorstore.save_local(vectorstore_path) # Save the vectorstore
        
    return vectorstore #, tools

vectorstore = setup_vectorstore() # Setup vectorstore
# Set up retriever
retriever = vectorstore.as_retriever()
multiquery_retriever = MultiQueryRetriever.from_llm(retriever, llm) # Multiquery for better performance

# Define agent functions
class RouteQuery(BaseModel):
    '''Route a user query to the most relevant path'''
    datasource: Literal["retrieve", "web_search", "direct_answer"] = Field(
        ...,
        description="Choose to route the user question to vectorstore retrieval, web search, or answer directly."
    )

def query_router():
    structured_llm_router = llm.with_structured_output(RouteQuery)
    system = """You are an expert at routing a user question to a vectorstore or web search or to direct answering a query.
    The vectorstore contains documents related to the robot, Centre for Innovation Building in IIT Madras and the clubs and competition teams present in it. Use the vectorstore for questions on these topics. If the query is about recent events that you do not have knowledge of, use web search. If it is a general query that can be answered directly, route do neither, ie, direct answer."""
    
    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )
    query_router = route_prompt | structured_llm_router
    return query_router

def rag_generator():
    prompt = PromptTemplate(template="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise unless you are specifically asked for details.
    Question: {question} 
    Context: {context} 
    Answer:
    """,
    input_variables = ["question", "context"]
    )
    # Chain
    rag_chain = prompt | llm | StrOutputParser()
    
    return rag_chain

def direct_answer():
    # Define the prompt for direct answering
    system_prompt = """You are an AI assistant providing direct answers to user questions. 
    Respond clearly and concisely based on the user's input. Use maximum of three sentences unless you are specifically asked for details."""
    
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "User question: \n\n {question}"),
        ]
    )
    return answer_prompt | llm | StrOutputParser()  # Chain for direct answering

# Web search tool
web_search_tool = TavilySearchResults(api_key= tavily_api_key, max_results=3, search_depth = 'basic') 

# Setup the chains

query_router_chain = query_router()
rag_chain = rag_generator()
direct_answer_chain = direct_answer()

# Functions

def retrieve_and_generate(question):
    documents = multiquery_retriever.invoke(question)
    return rag_chain.invoke({"context": documents, "question": question})


# Thread-safe dictionary
class SafeDict:
    def __init__(self):
        self._data = {}
        self._lock = Lock()
    
    def set(self, key, value):
        with self._lock:
            self._data[key] = value
    
    def get(self, key):
        with self._lock:
            return self._data.get(key)

# Worker functions remain unchanged
def route_worker(question, return_dict):
    route_result = query_router_chain.invoke({"question": question})
    return_dict.set('route', route_result.datasource)

def retrieve_worker(question, return_dict):
    try:
        answer = rag_chain.invoke({"context": multiquery_retriever.invoke(question), "question": question})
        return_dict.set('retrieve', answer)
    except:
        return_dict.set('retrieve', '')

def web_search_worker(question, return_dict):
    try:
        web_results = web_search_tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in web_results])
        answer = rag_chain.invoke({"context": web_results, "question": question})
        return_dict.set('web_search', answer)
    except:
        return_dict.set('web_search', '')

def direct_answer_worker(question, return_dict):
    try:
        answer = direct_answer_chain.invoke({"question": question})
        return_dict.set('direct_answer', answer)
    except:
        return_dict.set('direct_answer', '')

def query_agent(question):
    return_dict = SafeDict()

    # Start threading tasks
    route_thread = Thread(target=route_worker, args=(question, return_dict))
    retrieve_thread = Thread(target=retrieve_worker, args=(question, return_dict))
    web_search_thread = Thread(target=web_search_worker, args=(question, return_dict))
    direct_answer_thread = Thread(target=direct_answer_worker, args=(question, return_dict))

    # Start threads
    route_thread.start()
    retrieve_thread.start()
    web_search_thread.start()
    direct_answer_thread.start()

    # Wait for routing result
    route_thread.join()

    datasource = return_dict.get('route')  # Get route decision

    # Handle tasks based on routing result
    if datasource == "retrieve":
        retrieve_thread.join()
        web_search_thread.join(timeout=0)  # Cancel web search
        direct_answer_thread.join(timeout=0)  # Cancel direct answer
        return return_dict.get('retrieve')
    elif datasource == "web_search":
        web_search_thread.join()
        retrieve_thread.join(timeout=0)  # Cancel retrieve
        direct_answer_thread.join(timeout=0)  # Cancel direct answer
        return return_dict.get('web_search')
    else:  # direct_answer
        direct_answer_thread.join()
        retrieve_thread.join(timeout=0)  # Cancel retrieve
        web_search_thread.join(timeout=0)  # Cancel web search
        return return_dict.get('direct_answer')


if __name__ == "__main__":
    # Testing
    import time
    # Measure time taken for each type of workflow
    def measure(question):
        start = time.time()
        # response = user_query(question)
        response = query_agent(question)
        end = time.time()
        print(f"Time taken for question: {end-start}")
        print(response)
    
    question = "What is AI CLub?"
    measure(question)
