from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from utils import format_question
from tqdm import tqdm


def ask_llama3(question, tokenizer, model, sys_message=None, temperature=0.1, max_new_tokens=128, do_sample=True, ic_examples=None):
    # Create messages structured for the chat template
    if ic_examples:
        messages = [{"role": "system", "content": sys_message}] + ic_examples + [{"role": "user", "content": question}]
    else:
        messages = [{"role": "system", "content": sys_message}, {"role": "user", "content": question}]

    # Applying chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    if do_sample:
        outputs = model.generate(**inputs, max_new_tokens= max_new_tokens, use_cache=True, temperature=temperature,
                             pad_token_id=tokenizer.eos_token_id, do_sample=do_sample)
    else:
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True,
                                 pad_token_id=tokenizer.eos_token_id, do_sample=do_sample)

    # Extract and return the generated text, removing the prompt
    response_text = tokenizer.batch_decode(outputs)[0].strip()
    answer = response_text.split('<|im_start|>assistant')[-1].strip()
    return answer

def load_llama3_model(model_name, bnb_config, device_map='auto'):
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, trust_remote_code=True,
                                                 use_cache=False, device_map=device_map)

    return model

def run_on_data_set(df_dataset, question_format, tokenizer, model,
                    sys_message=None, temperature=0.1, max_new_tokens=128, n_examples=None, do_sample=True,
                    ic_examples=None):
    results = []
    for i in tqdm(range(len(df_dataset))):
        question = format_question(df_dataset.iloc[i], question_format=question_format)
        long_answer = ask_llama3(question, tokenizer=tokenizer, model=model, sys_message=sys_message,
                                 temperature=temperature, max_new_tokens=max_new_tokens, do_sample=do_sample,
                                 ic_examples=ic_examples)
        question_result = {"question": question, "long_answer": long_answer, "correct_answer": int(df_dataset.iloc[i].cop)}
        results.append(question_result)

        if n_examples:
            if i + 1 == n_examples:
                break

    return results
