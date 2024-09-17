
# Metadata Extraction from Research Papers

## Objective

This use case demonstrates how Aggrag can be used to extract metadata from research papers. The objective is to extract metadata **strictly** in the following format: 

{
"title":string, 
"authors":[list of strings],
"organizations":[list of strings],
"keywords":[list of strings]
}


## Usecase-Iteration Flow

Aggrag follows a Usecase-Iteration flow, where each use case consists of multiple iterations. Each iteration is designed to improve upon the previous one, either by refining the prompt, adjusting the RAG parameters, or incorporating additional data or techniques.

An integral part of an iteration is setting up the evaluation model. Although more experiments are to be done, intuitivally it makes sense to evaluate all iterations on same or similar metrics. 

In this example, we are evaluating generated responses' strict adherence to metadata schema and the time taken to extract. 

Note: ideally, adding more than one paper should not affect the time taken as aggrag library is designed to work asynchronously. But in the current example iterations, we are noticing some latency being added as more pdf papers are added. 

## Testing the Flow

You can get your hands dirty by setting up an example flow to learn the capabilities of what  the aggrag example flows by following these steps:

1. Setup and run the aggrag project locally. Steps here: `https://github.com/genai-apps/aggrag#local-setup`
2. Create a new 'use-case' on aggrag UI.
3. On iteration 1, import a sample cforge file located in examples/ directory. 'Import' is button on top-right banner on the UI.
4. To run the flow locally, you'll need to set up the aggrag environment and provide your model API keys in the .env file. Follow the instructions in the Local Setup section of the README.

    Note: You could also follow the on-screen hints on the UI.

###### Pre-requisites
1. AzureOpenAI or OpenAI keys. Each RAG has a settings' UI, where configuration such as models, prompts etc can be modified. 
2. PDF research papers. 1 sample pdf research paper is added in examples/metadata_extraction/


By testing the flow with your own research papers, you can explore the capabilities of Aggrag and potentially contribute to improving the metadata extraction use case.


### Iterations

#### Setup, Evlation, Inference and Conclusion 

In 3 iterations, we finalise the configurations for metadata extraction use case, and deploy in the final application.

**Evaluation**: All 3 iterations described below were tested on two parameters:
    1. Whether generated responses matched the metadata schema in the objective. 
    2. Time taken to extract the generated response

    The evaluation flow is also setup in the aggrag UI.

#### Iteration 1:

**Setup**: Two type of RAGs are used viz. Base and SubQA;

**Inference**: 
    1. Base RAG has 0% accuracy; SubQA has at best 65% accuracy 
    2. SubQA RAG takes significantly longer to run (between 30 and 70 seconds)

**Conclusion**: Existing RAG in the ragstore are not sufficient, need a new RAG.

#### Iteration 2:

**Setup**: A new RAG, MetaLang, was designed to extract metadata and then added to the RAGstore. Iteration 2 uses MetaLang and Subqa from iteration 1.

**Inference**: 
    1. Meta Lang has 100% accuracy with both gpt3.5 and gpt4. Ideal choice for our objective.
    2. Metalang take a lot of time to generate results, bottleneck for our use case.


**Conclusion**: It is possible to create a RAG with 100% accuracy, but needs to be faster for our particular use case.


#### Iteration 3:

**Setup**: Meta-Llama RAG was designed to improve on the speed of Meta Lang RAG without losing any accuracy. 

**Inference**: 
    1. Both Meta Lang and Meta Llama have 100% accuracy.
    2. No significant improvement observed in 


**Conclusion**: Good for now, but we can attempt using cheaper and smaller LLM models to reduce response time and costs. 


The above example is from a real world use case, where *Iteration 2 was used for a brief period before switching to iteration 3. Client preferences :]* 
