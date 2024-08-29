
# 🎛️ Aggrag 

### Built with

1. Python 3.11.9
2. Node.js 22.2.0

Aggrag allows you to configure LLM experiments, and take them from prototype to deployment wthin a single framework. With Aggrag library, you get a framework that allows:

- **Multiple RAGs are better than one RAG**: Configure experiments with an ensemble of RAG pipelines
- **Evaluate the quality of the generated outputs**: Evaluate the quality of the generated outputs across different RAG pipelines
- **RAGstore: Different RAGs for different tasks**: Configure different RAG pipelines for different use cases
- **BYOR: Bring your own RAG**: Aggrag allows you to plug in your own RAG pipelines, either permanently or for each experiment
- **UI and Framework**: Aggrag comes with a framework for development for jupyter notebooks etc, and a UI that allows you to easily configure and run experiments
- **Host Application compatibility**: Take your experiments to the next level by deploying them in your custom applications
- **Metrics driven iterative development**: Build iterations on your experiments based on the insights generated either from the real world users using your application, or from the insights from your own experiments

# Table of Contents

- [Local Setup](#local-setup)
- [Managing Dependencies](#managing-dependencies)
- [Documentation](#documentation)
- [Installation](#installation)
- [RAGstore](#ragstore)
- [Example Use Cases](#example-use-cases)
- [Features](#features) 
- [Citations](#citations)




# Local Setup

  1. import the repo from https://github.com/genai-apps/aggrag.git. checkout to 'develop' branch
  2. To start the UI: get into the 'react-server' directory, and run `npm install` and then `npm run start`
  3. To start the server: get into the root aggrag directory and run: `python -m library.app serve`. Note, however:
      - You will need to create `.env` file in the root directory. Use the `.env-sample` file as a reference.
      - run `pip install -r requirements.txt`. It is recommended to create a new virtualenv in the root directory to avoid installing packages globally. Command to create a new virtual env: `python3 -m venv venv`


# Managing Dependencies
 
When adding a new package, please make sure to:

 1. Add the package to the `requirements.in` file.
 2. Run `pip-compile requirements.in` to automatically update the `requirements.txt` file with the new package 
and its sub-dependencies.
 3. Use `pip install -r requirements.txt` to install the updated dependencies.

NOTE: After adding a new package and running `pip-compile`, always run `pip install -r requirements.txt` to ensure 
that all dependencies are installed correctly and that there are no issues with the installation process.


# Documentation
Detailed documentation is work in progress. 

# Installation

You will be able to install aggrag locally with Python 3.8 and higher, or use a playground environment hosted by us. The web version of aggrag will have a limited feature set. In a locally installed version you can load API keys automatically from environment variables, write Python code to evaluate LLM/RAG responses.

# RAGstore

Aggrag library comes with a directory of RAGs that you can integrate into your application on the fly, or run experiments with them. More details coming soon.

# Supported providers

- OpenAI
- Anthropic
- Google (Gemini, PaLM2)
- HuggingFace (Inference and Endpoints)
- [Ollama](https://github.com/jmorganca/ollama) (locally-hosted models)
- Microsoft Azure OpenAI Endpoints
- [AlephAlpha](https://app.aleph-alpha.com/)
- Foundation models via Amazon Bedrock on-demand inference, including Anthropic Claude 3
- ...and any other provider through [custom provider scripts]()!

# Example use cases

WIP

---

# License

Aggrag is released under the MIT License.

# Citations

Aggrag is our original concept, however, for implementation we have relied on the existing development of: [arXiv pre-print](https://arxiv.org/abs/2309.09128)
