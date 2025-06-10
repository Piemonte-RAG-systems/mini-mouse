import json, os
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine

os.environ["OPENAI_API_KEY"] = 'ollama'
configuration = json.load(open("config.json"))
Settings.llm = Ollama(model=configuration["language_model"])

def query(input_text):
  query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[],
  )
  response = query_engine.query(str(input_text))
  return str(response)

def query_simple(input_text):
  llm = Ollama( model=configuration["language_model"], 
               request_timeout=90.0)
  return str(llm.complete(input_text))

print('Hello! Welcome to the Mini Mouse app. Feel free to chat away and make sure to use "\bye" when you are finished.')

end_program = False

while not end_program:
  input_query = input()
  if input_query == '/bye':
    end_program = True
  else:
    print(query_simple(input_query))
