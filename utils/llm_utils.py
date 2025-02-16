import os
import json
import openai
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import ChatCohere


def instantiate_gpt4():
    openai.api_key = os.getenv("OPENAI_API_KEY")
    model = ChatOpenAI(
        temperature=temperature,
        model="gpt-4",
        max_tokens=max_tokens,
        openai_api_key=openai.api_key,
    )
    return model
    return model

def instantiate_cohere(temperature=0.2, max_tokens=2000):
    model = ChatCohere(
        temperature=temperature,
        max_tokens=max_tokens
    )
    return model

def instantiate_gemini_pro():
    temperature = 0
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature = temperature)
    return model

def instantiate_gemini_20_flash() :
    temperature = 1
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature = temperature)
    return model


def instantiate_claude_3_opus():
    temperature = 0
    model = ChatAnthropic(model='claude-3-opus-20240229', temperature = temperature, max_tokens=4000)
    return model

def instantiate_claude_3_sonnet():
    temperature = 0
    model = ChatAnthropic(model='claude-3-sonnet-20240229', temperature = temperature, max_tokens=4000)
    return model
    
def instantiate_claude_3_haiku():
    temperature = 0
    model = ChatAnthropic(model='claude-3-haiku-20240307', temperature = temperature, max_tokens=4000)
    return model

def instantiate_llm_studio(temperature=0.2, max_tokens=2000):
    openai.api_base = "http://127.0.0.1:1234/v1"
    model = ChatOpenAI(
        temperature=temperature,
        max_tokens=max_tokens,
        openai_api_base = "http://127.0.0.1:1234/v1"
    )
    return model

def instantiate_llm_model(model_name):
    switch_dict = {
        "cohere" : instantiate_cohere,
        "gemini_20_flash" : instantiate_gemini_20_flash,
        "gemini_pro" : instantiate_gemini_pro,
        "gpt4": instantiate_gpt4,
        "claude_3_opus" : instantiate_claude_3_opus,
        "claude_3_sonnet" : instantiate_claude_3_sonnet,
        "claude_3_haiku" : instantiate_claude_3_haiku,
        "llm_studio" : instantiate_llm_studio
    }
    # Get the function for the given model_name, or default if not found
    selected_case = switch_dict.get(model_name)
    # Execute the selected function and return the result
    return selected_case()

def make_llm_inference(invoke_parameters_dict, prompt_path, model_name) :

    model = instantiate_llm_model(model_name)


    with open(prompt_path, 'r') as file:
        prompt_json = json.load(file)

    prompt_list = []
    for item in prompt_json:
        for role, message in item.items():
            prompt_list.append((role, message))

    prompt   = ChatPromptTemplate.from_messages(prompt_list)

    chain = (
        prompt
        | model
    )
    try :
        response = chain.invoke(invoke_parameters_dict)
        return response
    except Exception as e:
        print(f"An error occurred: {e}")
        return None