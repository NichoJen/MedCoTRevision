import re
import json
import pandas as pd

def format_question(example, question_format="default_v1"):
    """
    :param example: question in MedMCQA format as example from pandas dataframe
    :type example:
    :param question_format:
    :type question_format:
    :return:
    :rtype:
    """
    if question_format == "default_v1":
        return f"Question: {example.question} Options: A) {example.opa} B) {example.opb} C) {example.opc} D) {example.opd}"

def extract_answer(llama_output):
    # Define the pattern to search for the answer
    pattern = r"answer is:?\s*([A-D])|answer is:\s*\n\n([A-D])|diagnosis is:?\s*([A-D])|diagnosis is:\s*\n\n([A-D])"


    # Search for the pattern in the model's output
    match = re.search(pattern, llama_output)

    if match:
        # Extract and return the first non-None group
        for i in range(1, 5):
            if match.group(i):
                return match.group(i)
    return None

def compute_metrics(outputs, conversion_dict):
  correct_examples = 0
  for example in outputs:
    if example["extracted_answer"]: # check if extracted answer is not none
      extracted_answer = conversion_dict[example["extracted_answer"]]
      if example["correct_answer"] == extracted_answer:
        correct_examples += 1
  return {"accuracy": correct_examples/len(outputs)}

def results_to_json_file(results, path):
    with open(path, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=4, sort_keys=True)

def load_in_context_examples(path_or_df, conversion_dict, example_ids = None):
    if isinstance(path_or_df, str):
        examples_df = pd.read_csv(path_or_df)
    else:
        examples_df = path_or_df
    if example_ids is None:
        example_ids = examples_df["id"]
    ic_examples = []
    for example_id in example_ids:
        example = examples_df.loc[examples_df['id'] == example_id].iloc[0]
        #print(example)
        question = format_question(example)
        #print("question: ", question)
        explanation = example.exp
        #print("explanation: ", explanation)
        cop = int(example.cop)
        cop_letter = conversion_dict[cop] # -> A-D
        answer_op_text = example["op" + cop_letter.lower()] # op + a-d

        ic_question = {"role": "user", "content": question}
        answer = f"{explanation} Therefore, the correct answer is {cop_letter}) {answer_op_text}"
        ic_answer = {"role": "assistant", "content": answer}

        ic_examples.append(ic_question)
        ic_examples.append(ic_answer)

    return ic_examples

def exp_rephrase_prompt(example, conversion_dict, cot_rephrase_prompt_p1, cot_rephrase_prompt_p2):
  question = format_question(example)
  explanation = example.exp
  cop = int(example.cop)
  cop_letter = conversion_dict[cop] # -> A-D
  answer_op_text = example["op" + cop_letter.lower()] # op + a-d
  full_answer = f"{explanation} Therefore, the answer is {cop_letter}) {answer_op_text}"
  full_rephrase_propmt = cot_rephrase_prompt_p1 + question + " Explanation: " + full_answer + cot_rephrase_prompt_p2
  return full_rephrase_propmt