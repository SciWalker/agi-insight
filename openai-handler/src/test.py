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
print(type(retrieved_data[4]))
print(retrieved_data[4]['content'])
#convert the content dictionary to string
string_data = json.dumps(retrieved_data[1])

#modify the content string to replace \n with \\n
for item in retrieved_data:
    item["content"] = item["content"]
    # item["content"] = json.loads(item["content"])
with open("../output/retrieved_data.txt", "w") as file:
    file.write(string_data)
# #save retrieved_data to a text file
# with open("../output/retrieved_data.txt", "w") as file:
#     #use json.dumps with indent=4
#     file.write(json.dumps(retrieved_data, indent=4))
#     print(json.dumps(retrieved_data, indent=4))