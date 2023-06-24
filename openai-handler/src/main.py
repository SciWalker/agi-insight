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
data= [   {
        "role": "system",
        "content": "you are one of the top full stack developers, you will help me write wonderful and efficient codes"
    },
    {
        "role": "user",
        "content": """
        Help me check my code:
               print(retrieved_data)
        formatted_response
        #save the message_list to a text file
        # with open(message_list_path, 'w') as f:
        #     for item in message_list:
        #       f.write("%s \n" % item)
        """
    }]

'''
choose the model:
gpt-3.5-turbo is the cheapest model
gpt-4 is the best model with higher cost
'''
chatgpt_script_res=openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=retrieved_data
)

response_text = chatgpt_script_res['choices'][0]['message']
print(f"response_text: {type(response_text)}")
print(response_text)
# Replace \n with actual new lines
formatted_response = response_text['content'].replace('\\n', '\n')

# Save the formatted content to a file
with open("../output/formatted_response.txt", "w") as file:
    file.write(formatted_response)

retrieved_data.append(response_text)

# Save the list to a JSON file
with open(message_list_path, "w") as file:
    file.write(json.dumps(retrieved_data, indent=4))

# Load the list from the JSON file
with open(message_list_path, "r") as file:
    retrieved_data = json.load(file)

print(retrieved_data)
formatted_response

#save the message_list to a text file
# with open(message_list_path, 'w') as f:
#     for item in message_list:
#       f.write("%s \n" % item)


