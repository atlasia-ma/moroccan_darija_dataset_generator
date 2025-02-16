import utils.llm_utils as llm_utils
import json
import os
import time
from datasets import load_dataset
from utils.config_utils import load_config, get_config

from dotenv import load_dotenv


config_path = "config/config.yaml"
config = load_config(config_path)

print("config : ",config)

load_dotenv()

model_to_use = get_config(config, "llms", "llm_name")
print("model_to_use : ",model_to_use)


prompt_path = "./prompts/prompt_text_generation.json"

extract_size = 1000
number_of_topics = 4
sleep_between_each_generation = True
sleep_time = 20

def build_invoke_parameters(x):
    """Build the prompt based on the generation type"""
    snippet = x["question_title"].strip()
    snippet = snippet[:min(len(snippet), extract_size)]
    return {"EXTRACT" : snippet}


def generate_data():
    ds = load_dataset("community-datasets/yahoo_answers_topics", split="train", num_proc=2)
    print("ds : ",ds)
    ds = ds.select(range(number_of_topics))
    print("ds : ",ds)
    folder_path = "data/darija_dataset"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for index, row in enumerate(ds):
        print("row :", row)
        invoke_parameters_dict = build_invoke_parameters(row)
        print(invoke_parameters_dict)
        response = llm_utils.make_llm_inference(invoke_parameters_dict, prompt_path, model_to_use)
        print(response)
        if response is not None :
            content = response.content
            print("content : ", content)
            with open(f"{folder_path}/result_{model_to_use}_{index}.txt", "w", encoding= "utf-8") as file:
                file.write(content)
        # On some LLM APIs there is a rate limit for the number of generated tokens / minute
        # We add a sleep betweeen our generations
        if sleep_between_each_generation :
            time.sleep(sleep_time)

generate_data()