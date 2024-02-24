from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, GenerationConfig
import torch
import torch.nn as nn
from datasets import load_dataset
from functools import partial
import matplotlib
# Set the backend to 'Agg' to avoid GUI-related errors
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
os.environ["WANDB_DISABLED"] = "true"
set_seed(42)
#check my cuda
print("is cuda available:",torch.cuda.is_available())

modelpath = "models/phi-2"
model = AutoModelForCausalLM.from_pretrained(modelpath)
model = model.half()  # Convert model to float16
# quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

tokenizer = AutoTokenizer.from_pretrained(modelpath, use_fast=False)
tokenizer.add_tokens(["", "<PAD>"])
tokenizer.pad_token = "<PAD>"
tokenizer.add_special_tokens(dict(eos_token=""))
model.config.eos_token_id = tokenizer.eos_token_id

from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True) 

lora_config = LoraConfig(
    r=32, 
    lora_alpha=32, 
    target_modules = [ "q_proj", "k_proj", "v_proj", "dense" ],
    modules_to_save = ["lm_head", "embed_tokens"],
    lora_dropout=0.1, 
    bias="none", 
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

model.config.use_cache = False

dataset = load_dataset("g-ronimo/riddles_evolved")
dataset = dataset["train"].train_test_split(test_size=0.1)
print(dataset["train"][0])

IGNORE_INDEX=-100

def tokenize(input, max_length):
    input_ids, attention_mask, labels = [], [], []
    templates=["assistant\n{msg}", "user\n{msg}"]
    for i, msg in enumerate(input["messages"]):
        isHuman = i % 2 == 0
        msg_chatml = templates[isHuman].format(msg=msg)
        msg_tokenized = tokenizer(msg_chatml, truncation=False, add_special_tokens=False)
        input_ids += msg_tokenized["input_ids"]
        attention_mask += msg_tokenized["attention_mask"]
        labels += [IGNORE_INDEX] * len(msg_tokenized["input_ids"]) if isHuman else msg_tokenized["input_ids"]
    return {"input_ids": input_ids[:max_length], "attention_mask": attention_mask[:max_length], "labels": labels[:max_length]}

dataset_tokenized = dataset.map(partial(tokenize, max_length=1024), batched=False, num_proc=1, remove_columns=dataset["train"].column_names)

data = [len(tok) for tok in (dataset_tokenized["train"]["input_ids"] + dataset_tokenized["test"]["input_ids"])]
print(f"longest sample: {max(data)} tokens")

plt.hist(data, bins=10)
plt.show()

def collate(elements):
    tokens = [e["input_ids"] for e in elements]
    tokens_maxlen = max([len(t) for t in tokens])
    for i, sample in enumerate(elements):
        input_ids = sample["input_ids"]
        labels = sample["labels"]
        attention_mask = sample["attention_mask"]
        pad_len = tokens_maxlen - len(input_ids)
        input_ids.extend(pad_len * [tokenizer.pad_token_id])
        labels.extend(pad_len * [IGNORE_INDEX])
        attention_mask.extend(pad_len * [0])
    batch = {"input_ids": torch.tensor([e["input_ids"] for e in elements]), "labels": torch.tensor([e["labels"] for e in elements]), "attention_mask": torch.tensor([e["attention_mask"] for e in elements])}
    return batch

bs = 1
bs_eval = 8
ga_steps = 64
lr = 0.00002
epochs = 20
steps_per_epoch = len(dataset_tokenized["train"]) // (bs * ga_steps)

args = TrainingArguments(output_dir="out", per_device_train_batch_size=bs, per_device_eval_batch_size=bs_eval, evaluation_strategy="steps", logging_steps=1, eval_steps=steps_per_epoch // 2, save_steps=steps_per_epoch, gradient_accumulation_steps=ga_steps, num_train_epochs=epochs, lr_scheduler_type="constant", optim="adamw_hf", learning_rate=lr, group_by_length=False, bf16=False, fp16=True, ddp_find_unused_parameters=False)

trainer = Trainer(model=model, tokenizer=tokenizer, args=args, data_collator=collate, train_dataset=dataset_tokenized["train"], eval_dataset=dataset_tokenized["test"])

trainer.train()

base_path = "models/phi-2"
adapter_path = "out/checkpoint-1880"
save_to = "trained_model"

modelpath = "microsoft/phi-2"
model = AutoModelForCausalLM.from_pretrained(
    modelpath,    
    device_map="auto",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    ),
    torch_dtype=torch.bfloat16,
    # FA2 does not work yet
    # attn_implementation="flash_attention_2",          
)
tokenizer = AutoTokenizer.from_pretrained(base_path)
tokenizer.add_tokens(["", "<PAD>"])
tokenizer.pad_token = "<PAD>"
tokenizer.add_special_tokens(dict(eos_token=""))
model.config.eos_token_id = tokenizer.eos_token_id

generation_config = GenerationConfig(max_new_tokens=100, temperature=0.7, top_p=0.1, top_k=40, repetition_penalty=1.18, do_sample=True, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)

model = PeftModel.from_pretrained(model, adapter_path)
model = model.merge_and_unload()

model.save_pretrained(save_to, safe_serialization=True, max_shard_size='4GB')
tokenizer.save_pretrained(save_to)
generation_config.save_pretrained(save_to)

model_path = "trained_model"
question = "Hello, who are you?"

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

messages = [{"role": "user", "content": question}]

input_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")

output_tokens = model.generate(input_tokens)
output = tokenizer.decode(output_tokens[0][len(input_tokens[0]):], skip_special_tokens=True)

print(output)

modelpath = "trained_model"
model = AutoModelForCausalLM.from_pretrained(modelpath, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(modelpath)

model.push_to_hub("g-ronimo/phi-2_riddles-evolved")
tokenizer.push_to_hub("g-ronimo/phi-2_riddles-evolved")
