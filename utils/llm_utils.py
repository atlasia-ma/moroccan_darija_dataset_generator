from .config_utils import get_config
import os
import json
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatAnyscale
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI


def instantiate_gpt4():
    temperature = 0
    max_tokens = 8000
    request_timeout = 240
    top_p = 0.95
    max_tokens = 2000
    frequency_penalty = 0
    presence_penalty = 0
    stop = None

    MODEL_API_KEY = get_config("MODEL_API_KEY", "GPT4")
    MODEL_API_BASE = get_config("MODEL_API_BASE", "GPT4")
    MODEL_API_VERSION = get_config("MODEL_API_VERSION", "GPT4")
    MODEL_API_DEPLOYMENT_NAME = get_config("MODEL_API_DEPLOYMENT_NAME", "GPT4")
    model = AzureChatOpenAI(
        azure_endpoint = MODEL_API_BASE,
        openai_api_key = MODEL_API_KEY,
        openai_api_type = 'azure',
        openai_api_version = MODEL_API_VERSION,
        deployment_name = MODEL_API_DEPLOYMENT_NAME,
        model_name = "gpt-4",
        temperature = temperature, 
        max_tokens = max_tokens,
        request_timeout = request_timeout,
        model_kwargs={ 
            "top_p" : top_p,
            "frequency_penalty" : frequency_penalty,
            "presence_penalty" : presence_penalty,
            "stop" : stop
        }
    )
    return model

def instantiate_gemini_pro():
    temperature = 0
    MODEL_API_KEY = get_config("GOOGLE_API_KEY", "GEMINIPRO")
    os.environ["GOOGLE_API_KEY"] = MODEL_API_KEY
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature = temperature)
    return model

def instantiate_mistral_7b_openorca():
    temperature = 0
    MODEL_API_KEY = get_config("MODEL_API_KEY", "ANYSCALE")
    MODEL_API_BASE = get_config("MODEL_API_BASE", "ANYSCALE")
    os.environ["ANYSCALE_API_KEY"] = MODEL_API_KEY
    os.environ["ANYSCALE_API_BASE"] = MODEL_API_BASE
    model = ChatAnyscale(model_name="Open-Orca/Mistral-7B-OpenOrca", temperature = temperature)
    return model

def instantiate_claude_3_opus():
    temperature = 0
    MODEL_API_KEY = get_config("MODEL_API_KEY", "CLAUDE")
    os.environ["ANTHROPIC_API_KEY"] = MODEL_API_KEY
    model = ChatAnthropic(model='claude-3-opus-20240229', temperature = temperature, max_tokens=4000)
    return model

def instantiate_claude_3_sonnet():
    temperature = 0
    MODEL_API_KEY = get_config("MODEL_API_KEY", "CLAUDE")
    os.environ["ANTHROPIC_API_KEY"] = MODEL_API_KEY
    model = ChatAnthropic(model='claude-3-sonnet-20240229', temperature = temperature, max_tokens=4000)
    return model
    
def instantiate_claude_3_haiku():
    temperature = 0
    MODEL_API_KEY = get_config("MODEL_API_KEY", "CLAUDE")
    os.environ["ANTHROPIC_API_KEY"] = MODEL_API_KEY
    model = ChatAnthropic(model='claude-3-haiku-20240307', temperature = temperature, max_tokens=4000)
    return model

def instantiate_llm_model(model_name):
    switch_dict = {
        "gemini_pro" : instantiate_gemini_pro,
        "gpt4": instantiate_gpt4,
        "mistral_7b_openorca": instantiate_mistral_7b_openorca,
        "claude_3_opus" : instantiate_claude_3_opus,
        "claude_3_sonnet" : instantiate_claude_3_sonnet,
        "claude_3_haiku" : instantiate_claude_3_haiku
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