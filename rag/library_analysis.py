import os
from llmware.library import Library
from llmware.retrieval import Query
from llmware.prompts import Prompt
from llmware.setup import Setup
from llmware.models import ModelCatalog
from llmware.configs import LLMWareConfig
import time
LLMWareConfig().set_active_db("sqlite")
library = Library().create_new_library("Test1234")
# sample_files_path = Setup().load_sample_files()

library.add_files(os.path.join("data","check-addon-verif"))
print(Library().get_all_library_cards())

# library.install_new_embedding(embedding_model_name="all-MiniLM-L6-v2", vector_db="faiss")
library.install_new_embedding(embedding_model_name="industry-bert-contracts", vector_db="faiss")

os.environ["TOKENIZERS_PARALLELISM"] = "false" # Avoid a HuggingFace tokenizer warning
query_results = Query(library).semantic_query("API", result_count=2)

print(query_results)
query_results = Query(library).semantic_query("selfie_record", result_count=2)
print(query_results)
embedded_text = ''
for qr in query_results:
   embedded_text += '\n'.join(qr['text'].split("\'\'"))


# check all of the models for performance

model_list = ["mistral",
             "phi",
             "llama2",
             ]


# adapted from the BLING demo
query = "what is the use of selfie_record?"
for model_name in model_list:
    t0 = time.time()
    ModelCatalog().register_ollama_model(model_name)
    print(f"\n > Loading Model: {model_name}...")
    prompter = Prompt().load_model(model_name)
    
    t1 = time.time()
    print(f"\n > Model {model_name} load time: {t1-t0} seconds")
    
    print(f"Query: {query}")
    output = prompter.prompt_main(query, context=embedded_text
                                 , prompt_name="default_with_context",temperature=0.0)
    
    llm_response = output["llm_response"].strip("\n")
    print(f"LLM Response: {llm_response}")
    print(f"LLM Usage: {output['usage']}")
    
    t2 = time.time()
    print(f"\nTotal processing time: {t2-t1} seconds")