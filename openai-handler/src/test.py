import os
import openai
#read the json file config.json
import json
import ast
with open('../config/config.json') as f:
  config = json.load(f)
#set the api key
openai.api_key = config['openai_api_key']
# message_list_path='../output/thread/message_list_2.json'
message_list_path='../output/thread/message_list_2.txt'
# get the message list from the text file


with open(message_list_path, "r") as file:
    retrieved_data = file.read()
# Modify the JSON data to fix the syntax error

retrieved_data = ast.literal_eval(retrieved_data)
print(f"retrieved_data: {retrieved_data}")