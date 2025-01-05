from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pandas as pd
import torch
from utils import results_to_json_file
from run_llama3 import run_on_data_set
import os.path
from huggingface_hub import login


# log into huggingface
access_token = "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"
login(token=access_token)

# model and tokenizer setup
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
device_map = 'auto'
#bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.float16,)
bnb_config = None
model = AutoModelForCausalLM.from_pretrained(model_name,quantization_config=bnb_config, trust_remote_code=True,use_cache=False,device_map=device_map)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# dataset path
path_prefix = os.path.dirname(__file__)
dataset_path = path_prefix + "/../data/medmcqa_subset.csv"
print("loading dataset from: ", dataset_path)

# load dataset
questions_df = pd.read_csv(dataset_path)

# set model run args
run_name="zero_shot_base_v1_quantization"
temperature = 0.1
max_new_tokens = 128
do_sample = False # temperature ignored if do_sample=False


sys_message = '''
   You are a medical expert tasked with providing the most accurate answers to medical questions. Make sure your answers 
   are correct and concise.
   '''
question_format = "default_v1"

# run on dataset
results = run_on_data_set(df_dataset=questions_df, question_format=question_format, sys_message=sys_message,
                          tokenizer=tokenizer, model=model, temperature=temperature, max_new_tokens=max_new_tokens)

# write results
results_to_json_file(results=results, path=path_prefix + "/../outputs/" + run_name + ".json")

