from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pandas as pd
import torch

from run_llama3 import ask_llama3
from utils import exp_rephrase_prompt
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
run_name="cot_ic_set_2_exp_"
temperature = 0.1
max_new_tokens = 1500
do_sample = False # temperature ignored if do_sample=False


sys_message = '''
   You are a medical expert tasked with providing the most accurate answers to medical questions. Make sure your answers 
   are correct and concise.
   '''
question_format = "default_v1"

#cot_rephrase_prompt_p1 = """Given the following question and explanation, reformulate the explanation to include step-by-step reasoning: """
#cot_rephrase_prompt_prompt_p2 = """  Reformulated Explanation: Let's think step by step. """

prompts = {
    "Sequential Reasoning Prompt": {
        "part1": "Given the following question and explanation, rewrite it to include a logical sequence of reasoning:",
        "part2": "\nReformulated Explanation: Let's break it down logically. "
    },
    "Step-by-Step Clarification Prompt": {
        "part1": "Reformulate the following question and explanation to clarify the reasoning process step by step: ",
        "part2": "\nReformulated Explanation: To clarify, let's go through it step by step. "
    },
    "Detailed Reasoning Prompt": {
        "part1": "Rewrite the given question and explanation to include detailed reasoning in a step-by-step manner: ",
        "part2": "\nReformulated Explanation: Here's the detailed reasoning. "
    },
    "Logical Breakdown Prompt": {
        "part1": "Reformulate the following question and explanation to break down the reasoning logically: ",
        "part2": "\nReformulated Explanation: Let's break this down logically. "
    },
    "Stepwise Explanation Prompt": {
        "part1": "Rewrite the question and explanation below to include a stepwise reasoning process: ",
        "part2": "\nReformulated Explanation: Let's think through this step by step. "
    }
}

label_to_answer = {0: "A", 1: "B", 2: "C", 3: "D"} # convert numeric answers to alphabetic answers

# iterate through ic cot examples sets
ic_dataset_path = path_prefix + f"/../data/cot_ic_set_2.csv"
print("loading In-context examples from: ", ic_dataset_path)
for key in prompts:
    ic_df = pd.read_csv(ic_dataset_path)
    for i ,ic_id in enumerate(ic_df["id"]):
        print(f"IC example id: {ic_id}")

#    run_name_ic_set_version = run_name + str(i)
#    print(f"Run name: {run_name_ic_set_version}")

        reformat_prompt = exp_rephrase_prompt(ic_df.iloc[i], label_to_answer, cot_rephrase_prompt_p1=prompts[key]["part1"],
                                              cot_rephrase_prompt_p2=prompts[key]["part2"])

        print(reformat_prompt)

        reformat_exp = ask_llama3(reformat_prompt, tokenizer, model, sys_message=sys_message, temperature=temperature,
                              do_sample=do_sample, max_new_tokens=max_new_tokens)

        ic_df.loc[i, "exp"] = reformat_exp


#    subset_ic_df = ic_df[ic_df["id"] == ic_id]


#    ic_examples = load_in_context_examples(path_or_df=subset_ic_df, conversion_dict=label_to_answer)

    results_path = path_prefix + "/../data/" + run_name + key + ".csv"


    print("writing results to: ", results_path)
    ic_df.to_csv(results_path)

    # write results
#    results_to_json_file(results=results, path=results_path)