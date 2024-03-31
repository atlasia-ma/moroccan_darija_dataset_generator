import utils.llm_utils as llm_utils
import json
prompt_path = "./prompts/prompt_en_ar_pairs.json"
invoke_parameters_dict = {"N_SENTENCES" : 2}
selected_llm = "mistral_7b_openorca"
response = llm_utils.make_llm_inference(invoke_parameters_dict, prompt_path, selected_llm)
print(response)
if response is not None :
    content = response.content
    print("content : ", content)
    with open(f"result_{selected_llm}.txt", "w", encoding= "utf-8") as file:
        file.write(content)