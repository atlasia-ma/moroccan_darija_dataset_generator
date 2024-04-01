import utils.llm_utils as llm_utils
import json
import os
import time
from datasets import load_dataset


prompt_path = "./prompts/prompt_text_generation.json"
selected_llm = "claude_3_haiku"
extract_size = 1000
number_of_topics = 4
sleep_between_each_generation = True
sleep_time = 2

def build_invoke_parameters(x):
    """Build the prompt based on the generation type"""
    snippet = x["question_title"].strip()
    snippet = snippet[:min(len(snippet), extract_size)]
    return {"EXTRACT" : snippet}


def generate_data():
    ds = load_dataset("yahoo_answers_topics", split="train", num_proc=36)
    ds = ds.select(range(number_of_topics))

    folder_path = "data"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for index, row in enumerate(ds):
        invoke_parameters_dict = build_invoke_parameters(row)
        print(invoke_parameters_dict)
        response = llm_utils.make_llm_inference(invoke_parameters_dict, prompt_path, selected_llm)
        print(response)
        if response is not None :
            content = response.content
            print("content : ", content)
            with open(f"{folder_path}/result_{selected_llm}_{index}.txt", "w", encoding= "utf-8") as file:
                file.write(content)
        # On some LLM APIs there is a rate limit for the number of generated tokens / minute
        # We add a sleep betweeen our generations
        if sleep_between_each_generation :
            time.sleep(sleep_time)

generate_data()