import utils.llm_utils as llm_utils
from utils.config_utils import load_config, get_config
from dotenv import load_dotenv
from datasets import load_dataset, Dataset


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


def generate_dataset():
    """Generates translation"""
    dataset = load_dataset("sentence-transformers/mldr", "en-triplet", split="train")

    # Checkpoint file
    checkpoint_file = "translated_rows.txt"

    # Load already translated indices
    try:
        with open(checkpoint_file, "r") as f:
            translated_indices = set(map(int, f.read().splitlines()))
    except FileNotFoundError:
        translated_indices = set()

    # Load existing dataset from the hub
    repo_id = "salaheddinealabouch/mldr_moroccan_darija"
    try:
        existing_dataset = load_dataset(repo_id, split="train")
        existing_data = existing_dataset.to_list()
    except:
        existing_data = []

    # Open checkpoint file in append mode
    with open(checkpoint_file, "a") as f_checkpoint:
        new_data = []
        
        for i, row in enumerate(dataset):
            if i in translated_indices:
                continue  # Skip already translated rows
            
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

            # Save progress
            f_checkpoint.write(f"{i}\n")
            f_checkpoint.flush()  # Ensure writing to disk
            
            # Save data to hub every 100 rows
            if len(new_data) >= 3:
                merged_data = existing_data + new_data  # Append to existing data
                new_dataset = Dataset.from_list(merged_data)
                new_dataset.push_to_hub(repo_id)
                
                existing_data = merged_data  # Update existing data reference
                new_data = []  # Reset new_data for the next chunk

        # Final push if there are any remaining rows
        if new_data:
            merged_data = existing_data + new_data
            new_dataset = Dataset.from_list(merged_data)
            new_dataset.push_to_hub(repo_id)




generate_dataset()

