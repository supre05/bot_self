+#rag_pll.py
+# Implementation of adaptive self -corrective RAG, use graph RAG later
+'''
+Adaptive self corrective rag is good but too slow
+removing self correction doubles speed
+async doesn tmake much diff
+implemented multiquery retriever
+need to look
+reranking
+knowledge graphs
+'''
+import os
+from dotenv import load_dotenv
+from langchain_openai import ChatOpenAI
+from langchain_community.document_loaders import PyPDFLoader, TextLoader
+from langchain_openai import OpenAIEmbeddings
+from langchain.retrievers.multi_query import MultiQueryRetriever
+from langchain_text_splitters import RecursiveCharacterTextSplitter
+from langchain.prompts import PromptTemplate
+from langchain_core.output_parsers import StrOutputParser
+from pydantic import BaseModel, Field
+from langchain.prompts import PromptTemplate
+from langchain_core.prompts import ChatPromptTemplate
+import time
+from langchain_community.vectorstores import FAISS
+from typing import Literal
+load_dotenv()
+openai_api_key = os.getenv("OPENAI_API_KEY")
+tavily_api_key = os.getenv("TAVILY_API_KEY")
+embeddings = OpenAIEmbeddings(model = "text-embedding-3-large" , api_key = openai_api_key)
+llm = ChatOpenAI(
+            model="gpt-4o-mini",
+            temperature=0,
+            max_tokens=1000,
+            timeout=None,
+            max_retries=2,
+            api_key=openai_api_key,
+            # other params...
+        )
+rag_files = ["sample.pdf"]
+
+def _index_docs():
+    # Load, chunk, and index the contents of the data
+    docs = []
+    for file_name in rag_files:
+        data_file_path = f"./Data/Database_files/{file_name}"
+        if file_name.endswith(".pdf"):
+            loader = PyPDFLoader(data_file_path)
+        elif file_name.endswith(".txt"):
+            loader = TextLoader(data_file_path)
+        docs.extend(loader.load())
+    # Split and chunk the text
+    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
+    splits = text_splitter.split_documents(docs)
+    vectorstore = FAISS.from_documents(documents = splits, embedding = embeddings)
+    return vectorstore
+
+def setup_vectorstore():
+
+    # memory = MemorySaver(
+    # )
+    try:
+        vectorstore = FAISS.load_local(".Data/vectorstore", embeddings=embeddings, allow_dangerous_deserialization=True)
+    except:
+        print('Creating a new vectorstore')
+        vectorstore = _index_docs()
+        vectorstore.save_local(".Data/vectorstore")
+
+    # retriever_tool = create_retriever_tool(retriever, "law_content_retriever", "Searches and returns relevant excerpts from the Law and History of India document.")
+    # tools = [retriever_tool]
+
+    # (Optional) Set up a multi-query retriever for better performance (check with latency)
+    # retriever = MultiQueryRetriever(retriever, llm)
+
+    return vectorstore #, tools
+
+vectorstore = setup_vectorstore()
+# Set up retriever
+retriever = vectorstore.as_retriever()
+multiquery_retriever = MultiQueryRetriever.from_llm(retriever, llm)
+class RouteQuery(BaseModel):
+    '''Route a user query to the most relevant path'''
+    datasource: Literal["retrieve", "web_search", "direct_answer"] = Field(
+        ...,
+        description="Choose to route the user question to vectorstore retrieval, web search, or answer directly."
+    )
+
+def query_router():
+    structured_llm_router = llm.with_structured_output(RouteQuery)
+    system = """You are an expert at routing a user question to a vectorstore or web search or to direct answering a query.
+    The vectorstore contains documents related to the robot, Centre for Innovation Building in IIT Madras and the clubs and competition teams present in it. Use the vectorstore for questions on these topics. If the query is about recent events that you do not have knowledge of, use web search. If it is a general query that can be answered directly, route do neither, ie, direct answer."""
+
+    route_prompt = ChatPromptTemplate.from_messages(
+        [
+            ("system", system),
+            ("human", "{question}"),
+        ]
+    )
+    query_router = route_prompt | structured_llm_router
+    return query_router
+
+def rag_generator():
+    # Prompt
+    # prompt = hub.pull("rlm/rag-prompt")
+    prompt = PromptTemplate(template="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise unless you are specifically asked for details.
+    Question: {question}
+    Context: {context}
+    Answer:
+    """,
+    input_variables = ["question", "context"]
+    )
+    # Chain
+    rag_chain = prompt | llm | StrOutputParser()
+
+    return rag_chain
+
+def direct_answer():
+    # Define the prompt for direct answering
+    system_prompt = """You are an AI assistant providing direct answers to user questions.
+    Respond clearly and concisely based on the user's input. Use maximum of three sentences unless you are specifically asked for details."""
+
+    answer_prompt = ChatPromptTemplate.from_messages(
+        [
+            ("system", system_prompt),
+            ("human", "User question: \n\n {question}"),
+        ]
+    )
+    return answer_prompt | llm | StrOutputParser()  # Chain for direct answering
+
+# Web search tool
+from langchain_community.tools.tavily_search import TavilySearchResults
+web_search_tool = TavilySearchResults(api_key= tavily_api_key, max_results=3, search_depth = 'basic')
+
+# Setup the chains
+query_router_chain = query_router()
+rag_chain = rag_generator()
+direct_answer_chain = direct_answer()
+
+def handle_web_search(question):
+    web_results = web_search_tool.invoke({"query": question})
+    return "\n".join([d["content"] for d in web_results])
+
+def retrieve_and_generate(question):
+    documents = multiquery_retriever.invoke(question)
+    return rag_chain.invoke({"context": documents, "question": question})
+
+import multiprocessing as mp
+import time
+def route_worker(question, return_dict):
+    # Simulate the query router logic
+    route_result = query_router_chain.invoke({"question": question})
+    return_dict['route'] = route_result.datasource
+
+def retrieve_worker(question, return_dict):
+    # Simulate retrieval and generation logic
+    answer = rag_chain.invoke({"context": multiquery_retriever.invoke(question), "question": question})
+    return_dict['retrieve'] = answer
+
+def web_search_worker(question, return_dict):
+    # Simulate web search logic
+    web_results = web_search_tool.invoke({"query": question})
+    web_results = "\n".join([d["content"] for d in web_results])
+    return_dict['web_search'] = web_results  # Capture the answer from web search
+
+def direct_answer_worker(question, return_dict):
+    # Simulate direct answering logic
+    answer = direct_answer_chain.invoke({"question": question})
+    return_dict['direct_answer'] = answer
+def workflow(question):
+
+    manager = mp.Manager()
+    return_dict = manager.dict()
+
+    # Start multiprocessing tasks
+    route_process = mp.Process(target=route_worker, args=(question, return_dict))
+    route_process.start()
+    retrieve_process = mp.Process(target=retrieve_worker, args=(question, return_dict))
+    web_search_process = mp.Process(target=web_search_worker, args=(question, return_dict))
+    direct_answer_process = mp.Process(target=direct_answer_worker, args=(question, return_dict))
+    retrieve_process.start()
+    web_search_process.start()  # Wait for the web search process to finish
+    direct_answer_process.start()
+
+
+    # Wait for routing result
+    route_process.join()  # Ensure we get the routing result
+    datasource = return_dict.get('route')
+
+    # Start the relevant task based on routing result
+    if datasource == "retrieve":
+        retrieve_process.join()  # Wait for the retrieve process to finish
+        web_search_process.terminate()
+        direct_answer_process.terminate()
+        return return_dict['retrieve']
+
+    elif datasource == "web_search":
+        web_search_process.join()
+        retrieve_process.terminate()
+        direct_answer_process.terminate()
+        return return_dict['web_search']
+
+    else:  # direct_answer
+        direct_answer_process.join()  # Wait for the direct answer process to finish
+        retrieve_process.terminate()
+        web_search_process.terminate()
+        return return_dict['direct_answer']
+
+
+
+# if __name__ == "__main__":
+#     # For testing
+#     import time
+#     # Measure time taken for each type of workflow
+#     def measure(question):
+#         start = time.time()
+#         # response = user_query(question)
+#         response = workflow(question)
+#         end = time.time()
+#         print(f"Time taken for question: {end-start}")
+#         print(response)
+
+
+    # question = "What is AI Club?"
+    # print('Rag')
+    # measure(question)
+    # response = workflow(question)