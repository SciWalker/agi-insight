import os
import openai
#read the json file config.json
import json
with open('config/config.json') as f:
  config = json.load(f)
#set the api key
openai.api_key = config['openai_api_key']
message_list_path='thread/message_list_2.json'
# get the message list from the text file


with open(message_list_path, "r") as file:
    retrieved_data = json.load(file)

print(retrieved_data)
chatgpt_script_res=openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  # model="gpt-4",
  messages=retrieved_data
)


response_text=chatgpt_script_res['choices'][0]['message']
print(f"response_text: {type(response_text)}")
# cleaned_content = response_text.replace('\r', '')

# Replace \n with actual new lines
formatted_response = response_text['content'].replace('\\n', '\n')

# Save the formatted content to a file
with open("formatted_response.txt", "w") as file:
    file.write(formatted_response)

retrieved_data.append(response_text)

# Convert single quotes to double quotes
json_data = json.dumps(retrieved_data, indent=4)

# Save the list to a JSON file
with open(message_list_path, "w") as file:
    file.write(json_data)

# Load the list from the JSON file
with open(message_list_path, "r") as file:
    retrieved_data = json.load(file)

print(retrieved_data)
formatted_response
#save the message_list to a text file
# with open(message_list_path, 'w') as f:
#     for item in message_list:
#       f.write("%s \n" % item)


