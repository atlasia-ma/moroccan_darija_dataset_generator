# Moroccan Darija Dataset Generator

## Introduction

This project explores the generation of synthetic Moroccan darija dataset with billions of tokens to emulate the capabilities of proprietary LLMs such as GPT4, Gemini 2.0 Pro, Claude, ...

## Getting Started



```bash
# Clone the repository
git clone https://github.com/Alabouchsalaheddine/moroccan_darija_dataset_generator.git

# create a conda environment
conda create --name mddg_env python=3.10

# activate the created conda environment
conda mddg_env activate 
# try source mddg_env activate if conda doesn't work

# Install requirements
pip install -r requirements.txt

```


## Scripts Overview

### Choosing an LLM
To select an LLM, edit the `config/config.yaml` file and choose one of the supported LLMs:
- `cohere`
- `gemini_20_flash`
- `gemini_pro`
- `gpt4`
- `mistral_7b_openorca`
- `claude_3_opus`
- `claude_3_sonnet`
- `claude_3_haiku`
- `llm_studio`


### `generate_classification_dataset.py`
Generates a synthetic Moroccan Darija dataset for text classification. This can be used, for example, to fine-tune the **ModernBERT** model.

#### Usage

```bash
python generate_classification_dataset.py
```

### `generate.py`
Generates a synthetic Moroccan Darija dataset on different topics using the **Yahoo Answers Topics** dataset as a reference.

#### Usage
```bash
python generate.py
```

## Contributing
Contributions are welcome! Please follow these steps:
1. Commit your changes
2. Push to the branch
3. Open a Pull Request.


