
"""This example demonstrates a basic contract analysis workflow run entirely on on a laptop
    using a RAG-finetuned small specialized instruct BLING model
"""


import os
import re
from llmware.prompts import Prompt, HumanInTheLoop
from llmware.setup import Setup
from llmware.configs import LLMWareConfig
from llmware.models import ModelCatalog


def contract_analysis_on_laptop (model_name):
    # Load the llmware sample files
    print (f"\n > Loading the llmware sample files...")
    sample_files_path = Setup().load_sample_files()
    contracts_path = os.path.join(sample_files_path,"Agreements")
    
    # query list
    query_list = {"executive employment agreement": "What are the name of the two parties?",
                  "base salary": "What is the executive's base salary?",
                  "governing law": "What is the governing law?"}

    print (f"\n > Loading model {model_name}...")

    prompter = Prompt().load_model(model_name)

    for i, contract in enumerate(os.listdir(contracts_path)):

        # exclude potential mac os created file artifact in folder path
        if contract != ".DS_Store":
            
            print("\nAnalyzing contract: ", str(i+1), contract)

            print("LLM Responses:")
            
            for key, value in query_list.items():

                # contract is parsed, text-chunked, and then filtered by topic key
                print("contract",contracts_path)
                print("contract",contract)
                source = prompter.add_source_document(contracts_path, contract, query=key)

                # calling the LLM with 'source' information from the contract automatically packaged into the prompt
                responses = prompter.prompt_with_source(value, prompt_name="just_the_facts", temperature=0.3)

                for r, response in enumerate(responses):
                    print(key, ":", re.sub("[\n]"," ", response["llm_response"]).strip())

                # We're done with this contract, clear the source from the prompt
                prompter.clear_source_materials()

    # Save jsonl report to jsonl to /prompt_history folder
    print("\nPrompt state saved at: ", os.path.join(LLMWareConfig.get_prompt_path(),prompter.prompt_id))
    prompter.save_state()

    #Save csv report that includes the model, response, prompt, and evidence for human-in-the-loop review
    csv_output = HumanInTheLoop(prompter).export_current_interaction_to_csv()
    print("csv output - ", csv_output)


if __name__ == "__main__":
    # use local cpu model
    ModelCatalog().register_ollama_model(model_name="mistral")
    
    
    print("here")
    contract_analysis_on_laptop("mistral")
