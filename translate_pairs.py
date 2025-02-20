import utils.llm_utils as llm_utils
from utils.config_utils import load_config, get_config
from dotenv import load_dotenv
from datasets import load_dataset, Dataset, DatasetDict
import os
import json
import time 

config_path = "config/config.yaml"
config = load_config(config_path)

print("config : ",config)

load_dotenv()

model_to_use = get_config(config, "llms", "llm_name")
print("model_to_use : ",model_to_use)


prompt_path = "./prompts/prompt_translation.json"



def push_to_hub() :
    # Define the folder containing JSON files
    DATA_FOLDER = "translated_data"
    HUB_DATASET = "atlasia/all-nli-moroccan-darija"

    # Collect all JSON data
    all_data = []
    for filename in os.listdir(DATA_FOLDER):
        if filename.endswith(".json"):
            filepath = os.path.join(DATA_FOLDER, filename)
            with open(filepath, "r", encoding="utf-8") as file:
                data = json.load(file)
                all_data.extend(data)

    # Create a Hugging Face dataset
    dataset = Dataset.from_list(all_data)

    dataset_dict = DatasetDict({"train": dataset})


    # Print dataset size and first 5 samples
    print(f"Dataset size: {len(dataset)}")
    print(dataset[:5])


    # Push the dataset to the Hugging Face Hub
    dataset_dict.push_to_hub(HUB_DATASET)


    
def generate_translation(sentence_to_translate): 
    invoke_parameters_dict = {"sentence_to_translate" : sentence_to_translate}
    print("model_to_use : ",model_to_use)
    print("invoke_parameters_dict : ",invoke_parameters_dict)
    while True:
        try:
            response = llm_utils.make_llm_inference(invoke_parameters_dict, prompt_path, model_to_use)
            if response is not None:
                content = response.content
                print("content :", content)
                return content
        except Exception:
            print("Rate limit exceeded. Waiting 30 seconds before retrying...")
            time.sleep(30)



def load_translated_indices(checkpoint_file):
    """Load translated indices from the checkpoint file."""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            return set(map(int, f.read().splitlines()))
    return set()

def reserve_indices(dataset, checkpoint_file, batch_size=20):
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
    dataset = load_dataset("sentence-transformers/all-nli", "pair-score", split="train")
    checkpoint_file = "pairs_translated_rows.txt"
    output_dir = "translated_data"
    os.makedirs(output_dir, exist_ok=True)
    
    translation_cache = {}  # Cache for storing translations
    
    while True:  # Loop until no more data to process
        reserved_indices = reserve_indices(dataset, checkpoint_file, batch_size=200)
        if not reserved_indices:  # No more new rows to process
            break  

       
        for i in reserved_indices:
            new_data = []
            row = dataset[i]
            
            # Check cache for existing translations
            sentence1 = row["sentence1"]
            sentence2 = row["sentence2"]
            
            if sentence1 in translation_cache:
                translated_sentence1 = translation_cache[sentence1]
                print(f"Translation for '{sentence1}' found in cache.")
            else:
                translated_sentence1 = generate_translation(sentence1)
                translation_cache[sentence1] = translated_sentence1
                
            if sentence2 in translation_cache:
                translated_sentence2 = translation_cache[sentence2]
                print(f"Translation for '{sentence2}' found in cache.")
            else:
                translated_sentence2 = generate_translation(sentence2)
                translation_cache[sentence2] = translated_sentence2
                

            score = row["score"]

            metadata = {
                "sentence1_en": row["sentence1"],
                "sentence2_en": row["sentence2"]
            }
            new_data.append({
                "sentence1": translated_sentence1,
                "sentence2": translated_sentence2,
                "score": score,
                "metadata": metadata
            })

            # Save to JSON file for each reserved index
            output_file = os.path.join(output_dir, f"translated_{i}.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(new_data, f, ensure_ascii=False, indent=4)

generate_dataset()

# push_to_hub()
