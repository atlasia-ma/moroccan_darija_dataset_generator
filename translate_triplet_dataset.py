import utils.llm_utils as llm_utils
from utils.config_utils import load_config, get_config
from dotenv import load_dotenv
from datasets import load_dataset
import os
import json

config_path = "config/config.yaml"
config = load_config(config_path)

print("config : ",config)

load_dotenv()

model_to_use = get_config(config, "llms", "llm_name")
print("model_to_use : ",model_to_use)


prompt_path = "./prompts/prompt_translation.json"


def generate_translation(sentence_to_translate): 
    invoke_parameters_dict = {"sentence_to_translate" : sentence_to_translate}
    print("model_to_use : ",model_to_use)
    response = llm_utils.make_llm_inference(invoke_parameters_dict, prompt_path, model_to_use)
    print(response)
    if response is not None :
        content = response.content
        print("content : ", content)
        return content


def load_translated_indices(checkpoint_file):
    """Load translated indices from the checkpoint file."""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            return set(map(int, f.read().splitlines()))
    return set()

def reserve_indices(dataset, checkpoint_file, batch_size=3):
    """Reserve a batch of indices by writing them to the checkpoint file before processing."""
    translated_indices = load_translated_indices(checkpoint_file)
    reserved_indices = []

    # Find new rows to process
    for i, row in enumerate(dataset):
        if i not in translated_indices and len(reserved_indices) < batch_size:
            reserved_indices.append(i)

    # Write reserved indices to file before processing
    if reserved_indices:
        with open(checkpoint_file, "a") as f_checkpoint:
            for index in reserved_indices:
                f_checkpoint.write(f"{index}\n")

    return reserved_indices  # Return the indices this script will process

def generate_dataset():
    """Generates translation"""
    dataset = load_dataset("sentence-transformers/mldr", "en-triplet", split="train")
    checkpoint_file = "triplet_translated_rows.txt"
    output_dir = "translated_data"
    os.makedirs(output_dir, exist_ok=True)
    
    while True:  # Loop until no more data to process
        reserved_indices = reserve_indices(dataset, checkpoint_file, batch_size=3)
        if not reserved_indices:  # No more new rows to process
            break  

        new_data = []
        for i in reserved_indices:
            row = dataset[i]

            # Perform translation
            translated_anchor = generate_translation(row["anchor"])
            translated_positive = generate_translation(row["positive"])
            translated_negative = generate_translation(row["negative"])

            metadata = {
                "anchor_en": row["anchor"],
                "positive_en": row["positive"],
                "negative_en": row["negative"]
            }
            new_data.append({
                "anchor": translated_anchor,
                "positive": translated_positive,
                "negative": translated_negative,
                "metadata": metadata
            })

            # Save to JSON file for each reserved index
            output_file = os.path.join(output_dir, f"translated_{i}.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(new_data, f, ensure_ascii=False, indent=4)

generate_dataset()

