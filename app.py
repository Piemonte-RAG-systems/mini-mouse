import json, os
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
import nest_asyncio
nest_asyncio.apply()

from RAG.rag_objects import RAG_System

#import logging
#logging.basicConfig(level=logging.INFO)

os.environ["OPENAI_API_KEY"] = 'ollama'
configuration = json.load(open("config.json"))
Settings.llm = Ollama(model=configuration["language_model"])

def query_simple(input_text):
  llm = Ollama( model=configuration["language_model"], 
               request_timeout=90.0)
  return str(llm.complete(input_text))

print('Hello! Welcome to the Mini Mouse app. Feel free to chat away and make sure to use "/bye" when you are finished.')

end_program = False
rag_mode = False

while not end_program:
  input_query = input()
  split_query = input_query.split(" ")
  potential_command = split_query[0]
  if potential_command == '/bye':
    end_program = True
  elif potential_command == '/upload':
    if not rag_mode:
      rag_system = RAG_System()
      rag_mode = True
    
    if len(split_query[1:])%2==1:
      print('You must upload files with both path and name or else the command cannot be executed!')
    else:
      for i in range(len(split_query[1::2])):
        rag_system.embed_doc(split_query[1+i*2] , split_query[2+i*2])
        print('uploaded',split_query[2+i*2])
  else:
    if rag_mode:
      print(rag_system.query(input_query))
    else:
      print(query_simple(input_query))
