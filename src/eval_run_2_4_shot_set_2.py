from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pandas as pd
import torch
from utils import results_to_json_file
from utils import load_in_context_examples
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

# In-Context examples path
ic_dataset_path = path_prefix + "/../data/candidate_cot_dataset.csv"
print("loading In-context examples from: ", ic_dataset_path)


# set model run args
run_name="cot_2_4_shot_set_2_"
temperature = 0.1
max_new_tokens = 1500
do_sample = False # temperature ignored if do_sample=False


sys_message = '''
   You are a medical expert tasked with providing the most accurate answers to medical questions. Make sure your answers 
   are correct and concise.
   '''
question_format = "default_v1"

label_to_answer = {0: "A", 1: "B", 2: "C", 3: "D"} # convert numeric answers to alphabetic answers

# iterate through ic cot examples sets
ic_dataset_path = path_prefix + f"/../data/cot_ic_set_2.csv"
print("loading In-context examples from: ", ic_dataset_path)
ic_df = pd.read_csv(ic_dataset_path)
ic_ids = ["f745ba65-3599-4d35-9857-7e14768455f9",
          "0c381920-1397-4043-ad88-f9686c751d52",
          "0593c2e0-741a-4350-a5ab-1a610056af6a",
          "714e860b-ff8b-42e0-b427-0c076081d548"]
for i, ic_id in enumerate(ic_ids):
    #print(f"IC example id: {ic_id}")



    run_name_ic_set_version = run_name + "0_" + str(i)
    print(f"Run name: {run_name_ic_set_version}")

    subset_ic_df = ic_df[ic_df["id"].isin(ic_ids[0:i+1])]



    ic_examples = load_in_context_examples(path_or_df=subset_ic_df, conversion_dict=label_to_answer)

    print("length of ic examples: ", len(ic_examples))
    print(ic_examples)

    # run on dataset
    results = run_on_data_set(df_dataset=questions_df, question_format=question_format, sys_message=sys_message,
                          tokenizer=tokenizer, model=model, temperature=temperature, max_new_tokens=max_new_tokens,
                          ic_examples=ic_examples)

    results_path = path_prefix + "/../outputs/" + run_name_ic_set_version + ".json"

    print("writing results to: ", results_path)

    # write results
    results_to_json_file(results=results, path=results_path)