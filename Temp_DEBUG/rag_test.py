# Implementation of adaptive self -corrective RAG, use graph RAG later
'''
Adaptive self corrective rag is good but too slow
removing self correction doubles speed
async doesn tmake much diff
implemented multiquery retriever
need to look
reranking
knowledge graphs
'''
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain import hub
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt import tools_condition
from langchain.prompts import PromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, Sequence, TypedDict, Literal
from langgraph.graph.message import add_messages
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langgraph.graph import END, StateGraph, START
import pprint
import asyncio
from typing import List
from langchain_core.prompts import ChatPromptTemplate
import io
from langchain.schema import Document
from langchain_core.runnables import RunnablePassthrough
import time
from langchain_community.vectorstores import FAISS
from langchain.globals import set_llm_cache
from typing import Literal
from langchain_community.cache import InMemoryCache
# from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
# from langchain_community.document_compressors.rankllm_rerank import RankLLMRerank


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
embeddings = OpenAIEmbeddings(model = "text-embedding-3-large" , api_key = openai_api_key)
llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=1000,
            timeout=None,
            max_retries=2,
            api_key=openai_api_key,
            # other params...
        )
set_llm_cache(InMemoryCache())

rag_files = ["sample.pdf"]

def _index_docs():
    # Load, chunk, and index the contents of the data
    docs = []
    for file_name in rag_files:
        data_file_path = f"./Data/Database_files/{file_name}"
        if file_name.endswith(".pdf"):
            loader = PyPDFLoader(data_file_path)
        elif file_name.endswith(".txt"):
            loader = TextLoader(data_file_path)
        docs.extend(loader.load())
    # Split and chunk the text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    splits = text_splitter.split_documents(docs)
    # Store the chunks in a vectorstore
    vectorstore = Chroma.from_documents(documents = splits, embedding = embeddings, persist_directory = "./Data/vectorstore")
    
    return vectorstore

def setup_vectorstore():

    # memory = MemorySaver(
    # )    
    try:
        vectorstore = FAISS.load_local(".Data/vectorstore", embeddings=embeddings, allow_dangerous_deserialization=True)
    except:
        print('Creating a new vectorstore')
        vectorstore = _index_docs()
        vectorstore.save_local(".Data/vectorstore")
    
    # retriever_tool = create_retriever_tool(retriever, "law_content_retriever", "Searches and returns relevant excerpts from the Law and History of India document.")
    # tools = [retriever_tool]

    # (Optional) Set up a multi-query retriever for better performance (check with latency)
    # retriever = MultiQueryRetriever(retriever, llm)
        
    return vectorstore #, tools

vectorstore = setup_vectorstore()
# Set up retriever
retriever = vectorstore.as_retriever()
multiquery_retriever = MultiQueryRetriever.from_llm(retriever, llm)
# compressor = RankLLMRerank(top_n=3, model="zephyr")
# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=compressor, base_retriever=retriever
# )


class RouteQuery(BaseModel):
    '''Route a user query to the most relevant path'''
    datasource: Literal["vectorstore", "web_search", "direct_answer"] = Field(
        ...,
        description="Choose to route the user question to vectorstore retrieval, web search, or answer directly."
    )

def query_router():
    structured_llm_router = llm.with_structured_output(RouteQuery)
    # Prompt
    # system = """You are an expert at routing a user question to a vectorstore or web search or to direct answering a query.
    # The vectorstore contains documents related to history, law and constitution of India. Use the vectorstore for questions on these topics. If the query is about recent events that you do not have knowledge of, use web search. If it is a general query that can be answered directly, route do neither, ie, direct answer."""
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

class GradeDocs(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description = "Documents are relevant to the question?, 'yes' or 'no'")

def retrieval_grader():
    structured_llm_grader = llm.with_structured_output(GradeDocs)
    # Prompt
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
    )
    retrieval_grader = grade_prompt | structured_llm_grader
    return retrieval_grader

def rag_generator():
    # Prompt
    # prompt = hub.pull("rlm/rag-prompt")
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

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )
def hallucination_grader():
    structured_llm_grader = llm.with_structured_output(GradeHallucinations)

    # Prompt
    system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ]
    )
    hallucination_grader = hallucination_prompt | structured_llm_grader
    return hallucination_grader

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )
def answer_grader():
    structured_llm_grader = llm.with_structured_output(GradeAnswer)

    # Prompt
    system = """You are a grader assessing whether an answer addresses / resolves a question \n 
        Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
        ]
    )

    answer_grader = answer_prompt | structured_llm_grader
    return answer_grader

def query_rewritter():
    # Prompt
    system = """You a question re-writer that converts an input question to a better version that is optimized \n 
        for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Formulate an improved question.",
            ),
        ]
    )
    query_rewriter = re_write_prompt | llm | StrOutputParser()
    return query_rewriter

# Web search tool
from langchain_community.tools.tavily_search import TavilySearchResults
web_search_tool = TavilySearchResults(api_key= tavily_api_key, max_results=3, search_depth = 'basic') 

# Setup the chains
query_router_chain = query_router()
retrieval_grader_chain = retrieval_grader()
rag_chain = rag_generator()
hallucination_grader_chain = hallucination_grader()
answer_grader_chain = answer_grader()
query_rewriter_chain = query_rewritter()
direct_answer_chain = direct_answer()

# Construct Graph

# Define state
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]

# Define Graph Flow
def rag_retrieve(state):
    question = state["question"]
    # Retrieval
    documents = multiquery_retriever.invoke(question)
    return {"documents": documents, "question": question}

def rag_generate(state):
    question = state["question"]
    documents = state["documents"]
    # Generate
    response = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": response}

def direct_generate(state):
    question = state["question"]
    # Generate
    response = direct_answer_chain.invoke({"question": question})
    return {"question": question, "generation": response}

def grade_documents(state):
    question = state["question"]
    documents = state['documents']
    # Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader_chain.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            filtered_docs.append(d)
        else:
            continue
    return {"documents": filtered_docs, "question": question}

def transform_query(state):
    question = state["question"]
    documents = state["documents"]
    # Rewrite
    response = query_rewriter_chain.invoke({"question": question})
    return {"documents": documents, "question": response}

def web_search(state):
    print('a')
    question = state["question"]
    # Search
    web_results = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in web_results])
    web_results = Document(page_content=web_results)
    # Generate
    response = rag_chain.invoke({"context": web_results, "question": question})
    return {"documents": web_results, "question": question, "generation": response}

# Edges
def route_query(state):
    # Route
    print('Roiute')
    response = query_router_chain.invoke({"question": state["question"]})
    route = response.datasource
    if route == "vectorstore":
        return "vectorstore"
    elif route == "web_search":
        print('a')
        return "web_search"
    else:
        return "direct_answer"
    
def decide_to_generate(state):
    question = state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # No relevant documents, so rewrite the query
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        return "rag_generate"

def check_results(state):
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    # Check
    score = hallucination_grader_chain.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score.binary_score
    if grade == "yes":
        score = answer_grader_chain.invoke(
            {"question": question, "generation": generation}
        )
        grade = score.binary_score
        if grade == "yes":
            return "useful"
        else:
            return "not useful"
    else:
        return "not supported"
    
def agent_workflow():
    workflow = StateGraph(GraphState)
    workflow.add_node("rag_retrieve", rag_retrieve)
    workflow.add_node("web_search", web_search)
    workflow.add_node("direct_generate", direct_generate)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("rag_generate", rag_generate)
    workflow.add_node("grade_documents", grade_documents)

    workflow.add_conditional_edges(
        START,
        route_query,
        {
            "vectorstore": "rag_retrieve",
            "web_search": "web_search",
            "direct_answer": "direct_generate",
        },
    )
    workflow.add_edge("web_search", END)
    workflow.add_edge("direct_generate", END)
    workflow.add_edge("rag_retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "rag_generate": "rag_generate",
            "transform_query": "transform_query",
        },
    )
    workflow.add_edge("transform_query", "rag_retrieve")
    workflow.add_conditional_edges(
        "rag_generate",
        check_results,
        {
            "useful": END,
            "not useful": "transform_query",
            "not supported": "transform_query",
        },
    )
    graph = workflow.compile()
    # from PIL import Image

    # img = io.BytesIO(graph.get_graph().draw_mermaid_png())
    # img = Image.open(img)
    # img.save("rag_graph.png")
    return graph
agent = agent_workflow()

async def user_query(question):
    response = agent.invoke(
        {"question": question}
    )
    return response['generation']

    # for output in agent.stream({"messages": [HumanMessage(content=question)]}, stream_mode='updates'):
    #         for node, values in output.items():
    #             print(f"Node: {node}")
    #             print(values)
    #             print('\n\n')
    return None

# # Clean up function
# # def cleanup():
#     # vectorstore.delete_collection()

if __name__ == "__main__":
    # For testing
    # question = input("Enter your question: ")
    import time
    # Measure time taken for each type of workflow
    def measure(question):
        start = time.time()
        # response = user_query(question)
        response = asyncio.run(user_query(question))
        end = time.time()
        print(f"Time taken for question: {end-start}")
        print(response)
    
    # question = 'Why did Mira Murati leave openai?'
    # print('Search')
    # measure(question)
    # question = 'Hi, I am feeling happy today'
    # print('Direct')
    # measure(question)
 
    question = "What is the AI Club?"
    measure(question)
    # response = asyncio.run(user_query(question))
    # print(response)
    # cleanup()