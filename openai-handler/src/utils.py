# chatgpt_trans_res=openai.ChatCompletion.create(
#   model="gpt-3.5-turbo",
#   messages=[
#   {"role": "system", "content": "You are a helpful assistant that translates English to French."},
#   {"role": "user", "content": 'Translate the following English text to French: "{text}"'}
#     ]
# )
# instruct_curie=openai.Completion.create(
#   engine="text-curie-001",
#   prompt=prompt_1,
#   max_tokens=1800
# )