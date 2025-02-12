import os
import google.generativeai as genai
from google.api_core import exceptions
import random
import time

import csv
from bs4 import BeautifulSoup

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


                

os.environ["GEMINI_API_KEY"] = "YOUR GEMINI API KEY"
genai.configure(api_key=os.environ["GEMINI_API_KEY"])


def make_llm_inference(label, number_of_sentences_by_call):
    

    # Create the model
    generation_config = {
    "temperature": 2,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
    model_name="gemini-2.0-pro-exp-02-05",
    generation_config=generation_config,
    )

    chat_session = model.start_chat(
    history=[
        {
        "role": "user",
        "parts": [
            f"You are an expert linguist specializing in Moroccan Darija (الدارجة المغربية), with deep knowledge of its colloquial expressions and regional variations.\n\nGenerate {number_of_sentences_by_call} authentic, everyday Moroccan Darija sentences in Arabic script related to {label}. Focus on natural, conversational language as used by native speakers.\n\nRequirements:\n- Write each sentence in <sentence> tags\n- Use only Arabic script\n- Include common Darija expressions and idioms where appropriate\n- Vary between formal and informal speech patterns\n- Ensure sentences reflect authentic Moroccan speech patterns\n\nPlease provide only the sentences without translations or additional commentary."
        ]
        }
    ]
    )

    response = chat_session.send_message("INSERT_INPUT_HERE")
    print(response.text)
    return response


def generate_text():
    """Generates text in one of 26 domain classes"""
    number_of_sentences_by_call = 50
    counter = 0
    max_retries = 1000
    base_delay = 2  # Base delay in seconds

    labels = ['Adult', 'Arts_and_Entertainment', 'Autos_and_Vehicles', 'Beauty_and_Fitness', 'Books_and_Literature', 'Business_and_Industrial', 'Computers_and_Electronics', 'Finance', 'Food_and_Drink', 'Games', 'Health', 'Hobbies_and_Leisure', 'Home_and_Garden', 'Internet_and_Telecom', 'Jobs_and_Education', 'Law_and_Government', 'News', 'Online_Communities', 'People_and_Society', 'Pets_and_Animals', 'Real_Estate', 'Science', 'Sensitive_Subjects', 'Shopping', 'Sports', 'Travel_and_Transportation']
    for label in labels:
        for i in range (100) :
            success = False
            attempt = 0
            while not success and attempt < max_retries:
                try:
                    # Rate limiting check
                    if counter % 25 == 0 and counter != 0:
                        print("Rate limit prevention pause...")
                        time.sleep(60)  # Standard cool-down period

                    response = make_llm_inference(label, number_of_sentences_by_call)

                    # Check if there are any candidates and get the first one
                    if response.candidates:
                        first_candidate = response.candidates[0]
                        
                        # Check finish_reason
                        finish_reason = getattr(first_candidate, 'finish_reason', None)
                        
                        if finish_reason is not None and (
                                (isinstance(finish_reason, str) and finish_reason.upper() != "STOP") or
                                (isinstance(finish_reason, int) and finish_reason != 1)  #STOP enum value is 1 as an integer
                            ):
                            print(f"response: {response}")
                            print(f"first_candidate: {first_candidate}")
                            print(f"first_candidate.finish_reason: {first_candidate.finish_reason}")
                            print(f"Skipping due to finish reason: {first_candidate.finish_reason} and safety_ratings: {first_candidate.safety_ratings if hasattr(first_candidate, 'safety_ratings') else 'N/A'}")
                            success = True  # Skip this
                            break
                        else:
                            # Process successful response
                            html_string = response.text
                            sentences = parse_html_string(html_string)
                            # Write to a CSV file
                            write_to_csv(f'{label}.csv', sentences)
                    else:
                            print(f"Skipping due to no candidates in the response")
                            success = True
                            break


                    success = True
                    counter += 1
                    print(f"Successfully generated")

                except exceptions.ResourceExhausted as e:
                    attempt += 1
                    if attempt == max_retries:
                        print(f"Failed to generate after {max_retries} attempts")
                        raise

                    # Exponential backoff with jitter
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"Resource exhausted, attempt {attempt}/{max_retries}. Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)

                except Exception as e:
                    print(f"Unexpected error: {str(e)}")
                    attempt += 1
                    if attempt == max_retries:
                        raise
                    time.sleep(base_delay * (2 ** attempt))

generate_text()
