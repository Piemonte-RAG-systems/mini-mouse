import json, os
import nest_asyncio
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool

nest_asyncio.apply()
configuration = json.load(open("config.json"))
Settings.llm = Ollama(model=configuration["language_model"] , request_timeout=330.0)
Settings.embed_model = OllamaEmbedding(model_name=configuration["embedding_model"])
custom_llm = Ollama(
    model=configuration["language_model"],
    request_timeout=330.0
)

# To test embed_doc use the following command
# /upload ../Experiment_E_Lucas.pdf shear_modulus_report

class RAG_System:
    def __init__(self):
        self.tool_collection = {"summary_tools":[], "vector_tools":[]}
    def embed_doc(self, path_to_file, name_of_file):
        # load documents
        documents = SimpleDirectoryReader(input_files=[path_to_file]).load_data()
        
        # split the document into chunks (large chunk size for small docs)
        splitter = SentenceSplitter(chunk_size=4096)
        nodes = splitter.get_nodes_from_documents(documents)
        """
        this highlights how many sources (or chunks) were used
        print(len(nodes))
        """
        # setup different indexing systems with different methods
        summary_index = SummaryIndex(nodes)
        vector_index = VectorStoreIndex(nodes)
        
        # use only simple_summarize for speed
        summary_query_engine = summary_index.as_query_engine(
            response_mode="simple_summarize",
            use_async=False,
        )
        vector_query_engine = vector_index.as_query_engine()
        
        # add to the tool collections existing in the RAG system
        self.tool_collection["summary_tools"].append(QueryEngineTool.from_defaults(
            query_engine=summary_query_engine,
            description=f"Useful for summarization questions related to document called {name_of_file}",
        ))
        self.tool_collection["vector_tools"].append( QueryEngineTool.from_defaults(
            query_engine=vector_query_engine,
            description=f"Useful for retrieving specific context from the document called {name_of_file}",
            ))
        # just to note for above, it is good to provide the name of the file in case it cannot be infered
        # by the path

    def query(self, question_or_summary):
        query_engine = RouterQueryEngine(
            selector=LLMSingleSelector.from_defaults(llm=custom_llm),
            query_engine_tools=self.tool_collection["summary_tools"] + self.tool_collection["vector_tools"],
            verbose=False
    )
        response = query_engine.query(str(question_or_summary))
        return str(response)