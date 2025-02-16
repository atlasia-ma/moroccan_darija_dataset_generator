import utils.llm_utils as llm_utils
import os
import time
import os
import time
import csv
from bs4 import BeautifulSoup
from utils.config_utils import load_config, get_config
from dotenv import load_dotenv


config_path = "config/config.yaml"
config = load_config(config_path)

print("config : ",config)

load_dotenv()

model_to_use = get_config(config, "llms", "llm_name")
print("model_to_use : ",model_to_use)

def parse_html_string(html_string):
    # Parse the HTML string
    soup = BeautifulSoup(html_string, 'html.parser')

    # Collect the content inside each <sentence> tag
    sentences = []
    for tag in soup.find_all('sentence'):
        if tag.text:
            sentences.append(tag.text.strip())
    return sentences

def write_to_csv(filename, data):
    # Open the file in append mode
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = ['Text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write column names only if the file is empty
        if csvfile.tell() == 0:  # Check if the file is empty
            writer.writeheader()

        # Write each sentence's text content to the CSV file
        for sentence in data:
            writer.writerow({'Text': sentence})



prompt_path = "./prompts/prompt_classification_dataset.json"
delay = 20
folder_path = "data/classification_dataset"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

def generate_label_data(label, number_of_sentences_by_call): 
    invoke_parameters_dict = {"number_of_sentences_by_call" : number_of_sentences_by_call, "label" : label}
    print("invoke_parameters_dict : ",invoke_parameters_dict)
    print("prompt_path : ",prompt_path)
    print("model_to_use : ",model_to_use)
    response = llm_utils.make_llm_inference(invoke_parameters_dict, prompt_path, model_to_use)
    print(response)
    if response is not None :
        content = response.content
        print("content : ", content)
        return content



def generate_dataset():
    """Generates text in one of 26 domain classes"""
    number_of_sentences_by_call = 50
    labels = ['Adult', 'Arts_and_Entertainment', 'Autos_and_Vehicles', 'Beauty_and_Fitness', 'Books_and_Literature', 'Business_and_Industrial', 'Computers_and_Electronics', 'Finance', 'Food_and_Drink', 'Games', 'Health', 'Hobbies_and_Leisure', 'Home_and_Garden', 'Internet_and_Telecom', 'Jobs_and_Education', 'Law_and_Government', 'News', 'Online_Communities', 'People_and_Society', 'Pets_and_Animals', 'Real_Estate', 'Science', 'Sensitive_Subjects', 'Shopping', 'Sports', 'Travel_and_Transportation']
    for label in labels:
        for i in range (1) :
                try:
                    html_string = generate_label_data(label, number_of_sentences_by_call)
                    sentences = parse_html_string(html_string)
                    # Write to a CSV file
                    write_to_csv(f'{folder_path}/{label}.csv', sentences)
                    time.sleep(delay)

                except Exception as e:
                    print(f"Unexpected error: {str(e)}")

generate_dataset()

