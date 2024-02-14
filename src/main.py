import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_default_device("cuda")

model = AutoModelForCausalLM.from_pretrained("../models/phi-2", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("../models/phi-2", trust_remote_code=True)

inputs = tokenizer(["You are an expert muslim. Write the history of Macbook in the style of quran chapter 1","writer:"], return_tensors="pt", return_attention_mask=True)

# Adjust generate parameters to mitigate repetition
outputs = model.generate(**inputs, padding=True,max_length=200, num_beams=2,do_sample=True,early_stopping=True, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.8)

text = tokenizer.batch_decode(outputs)[0]

print(text)