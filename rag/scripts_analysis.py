"""This example demonstrates a basic file_to_be_analyzed analysis workflow run entirely on on a laptop
    using a RAG-finetuned small specialized instruct BLING model
"""
import os
import re
from llmware.prompts import Prompt, HumanInTheLoop
from llmware.setup import Setup
from llmware.configs import LLMWareConfig
from llmware.models import ModelCatalog


def analysis_on_laptop (model_name,fp,file_to_be_analyzed):




    query_list = {"code": "based on your study on this code,how was the code aesthetics?",
                    "bugs": "based on your study on this code,any potential bugs?",
                  "simplification": "based on your study on this code,which areas can it be simplified?"}


    if file_to_be_analyzed != ".DS_Store":
        
        print("\nAnalyzing file_to_be_analyzed: ", file_to_be_analyzed)

        print("LLM Responses:")
        prompter = Prompt().load_model(model_name)
        for key, value in query_list.items():
            flag_redundant_filename=False
            # file_to_be_analyzed is parsed, text-chunked, and then filtered by topic key
            print("file_to_be_analyzed",file_to_be_analyzed)
            if not file_to_be_analyzed.endswith(".pdf") and not file_to_be_analyzed.endswith(".txt") and not file_to_be_analyzed.endswith(".docx"):
                
                with open(os.path.join(fp,file_to_be_analyzed), 'r', encoding='utf-8') as python_file:
                    python_code = python_file.read()
                file_to_be_analyzed = file_to_be_analyzed.split(".")[0] + ".txt"
                #save the new file name
                new_file_to_be_written = os.path.join(fp,file_to_be_analyzed)
                with open(new_file_to_be_written, 'w', encoding='utf-8') as text_file:
                    text_file.write(python_code)


                flag_redundant_filename=True



            if file_to_be_analyzed.endswith(".pdf") or file_to_be_analyzed.endswith(".txt"):
                source = prompter.add_source_document(fp, file_to_be_analyzed,query="")
                # calling the LLM with 'source' information from the file_to_be_analyzed automatically packaged into the prompt
                responses = prompter.prompt_with_source(value, prompt_name="just_the_facts", temperature=0.3)
                if flag_redundant_filename==True:
                    os.remove(file_to_be_analyzed)
                for r, response in enumerate(responses):
                    print(key, ":", re.sub("[\n]"," ", response["llm_response"]).strip())

                # We're done with this file_to_be_analyzed, clear the source from the prompt
                prompter.clear_source_materials()

    # Save jsonl report to jsonl to /prompt_history folder
    print("\nPrompt state saved at: ", os.path.join(LLMWareConfig.get_prompt_path(),prompter.prompt_id))
    prompter.save_state()

    #Save csv report that includes the model, response, prompt, and evidence for human-in-the-loop review
    csv_output = HumanInTheLoop(prompter).export_current_interaction_to_csv()
    print("csv output - ", csv_output)

def search_for_files(path):
    list_files={}
    for root, dirs, files in os.walk(path):
        for file in files:
            # exclude DS_Store file
            if file != ".DS_Store":

                list_files[root]=file
    return list_files
if __name__ == "__main__":
    # use local cpu model
    ModelCatalog().register_ollama_model(model_name="mistral")
    
    list_path=search_for_files("data")
    print(list_path)
    for key in list_path:
        print(key)
        analysis_on_laptop("mistral",key,list_path[key])
