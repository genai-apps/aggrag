from ragas import evaluate


class Evaluator:

    """
    A class for evaluating language model responses against a set of metrics and benchmarks.

    Attributes:
        documents (iterable): A collection of documents used for creating the vector index.
        service_context (ServiceContext): The context configuration for service-related operations.
        llm_model (AzureOpenAI or other): The language model used for generating responses.
        llm_embeddings (AzureOpenAIEmbedding or other): The embedding model used for vectorizing text.
        rag_name (str): A unique name for the evaluation run, used for naming output files.
        project_path (str): The base path for storing evaluation reports and other artifacts.
        model_name (str): The name of the model configuration, appended to 'rag_name' for file naming.

    Methods:
        generate_responses: Generates responses from the language model for a list of questions.
        aevaluate_models: Asynchronously evaluates the model responses against specified metrics.
        evaluate_models: Synchronously evaluates the model responses against specified metrics.
        compare_llms: Compares the evaluation results of two Evaluator instances.
    """

    def __init__(self, response_dataset,
                 llm_model, 
                 llm_embeddings,
                 metrics):
        self.response_dataset = response_dataset
        self.llm_model = llm_model
        self.llm_embeddings = llm_embeddings
        self.metrics = metrics

    # TODO: Evaluate for per question answer pair

    async def aevaluate_models(self):

        """
        Asynchronously evaluates the generated responses using specified metrics.

        Parameters:
            response_dataset (Dataset, optional): A pre-generated dataset to evaluate.

        Returns:
            DataFrame: A pandas DataFrame containing the evaluation results.
        """

        result = evaluate(self.response_dataset, 
                          metrics=self.metrics,
                          is_async=True,
                          max_workers=4,
                          llm=self.llm_model, 
                          embeddings=self.llm_embeddings)
        result_df = result.to_pandas()
        return result_df

    def evaluate_models(self):
        """
        Synchronously evaluates the generated responses using specified metrics.

        Parameters:
            response_dataset (Dataset, optional): A pre-generated dataset to evaluate.

        Returns:
            DataFrame: A pandas DataFrame containing the evaluation results.
        """

        result = evaluate(self.response_dataset, 
                          metrics=self.metrics, 
                          llm=self.llm_model, 
                          embeddings=self.llm_embeddings)
        result_df = result.to_pandas()
        
        #result_df.to_csv("ll.csv", mode='w', index=False)
        return result_df


if __name__ == "__main__":

    #---------------------Generating Eval Matrices and  Metric Eval Report using two llm models------------------
    pass