from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms.azure_openai import AzureOpenAI
# from llama_index.llms.openai import OpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
    answer_similarity,
)
from ragas.metrics.critique import harmfulness
# from apps.evals.test_question_answers import test_questions, test_answers
# from apps.core.config import settings, AzureOpenAIModelNames, AzureOpenAIModelEngines

import os
import openai
import pandas as pd

# openai.api_key = OPENAI_API_KEY
# os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY


# azure_configs = {
#     "base_url": settings.AZURE_API_BASE,
#     "model_deployment": AzureOpenAIModelEngines.gpt_35_turbo_16k.value,
#     "model_name": AzureOpenAIModelNames.gpt_35_turbo_16k.value,
#     "embedding_deployment": AzureOpenAIModelEngines.text_embedding_ada_002.value,
#     "embedding_name": AzureOpenAIModelNames.text_embedding_ada_002.value,  # most likely
#     "openai_api_version": settings.OPENAI_API_VERSION
# }
#
# from langchain_openai.chat_models import AzureChatOpenAI
# from langchain_openai.embeddings import AzureOpenAIEmbeddings
#
# judge_azure_model = AzureChatOpenAI(
#     openai_api_version=azure_configs["openai_api_version"],
#     azure_endpoint=azure_configs["base_url"],
#     api_key=settings.AZURE_OPENAI_KEY,
#     azure_deployment=azure_configs["model_deployment"],
#     model=azure_configs["model_name"],
#     validate_base_url=False,
# )
#
# # init the embeddings for answer_relevancy, answer_correctness and answer_similarity
# judge_azure_embeddings = AzureOpenAIEmbeddings(
#     openai_api_version=azure_configs["openai_api_version"],
#     azure_endpoint=azure_configs["base_url"],
#     api_key=settings.AZURE_OPENAI_KEY,
#     azure_deployment=azure_configs["embedding_deployment"],
#     model=azure_configs["embedding_name"],
# )
# from apps.evals.utils import copy_sheet, copy_named_range
# import openpyxl


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

    def __init__(self, documents,
                 service_context,
                 llm_model,
                 llm_embeddings,
                 rag_name: str = None,
                 project_path: str = None,
                 model_name: str = None):
        self.documents = documents
        self.service_context = service_context
        self.llm_model = llm_model
        self.llm_embeddings = llm_embeddings
        self.project_path = project_path
        self.rag_name = f"{rag_name}_{model_name}"

    # TODO: Evaluate for per question answer pair

    def generate_responses(self, test_questions, test_answers=None):

        """
        Generates a dataset of responses for a given set of test questions using the configured language model.

        Parameters:
            test_questions (list): A list of strings, each representing a question to be queried to the language model.
            test_answers (list, optional): A list of expected answers for the test questions, used for comparison.

        Returns:
            Dataset: A 'datasets.Dataset' containing the questions, generated answers, contexts, and optionally the ground truths.
        """

        vector_index = VectorStoreIndex.from_documents(
            self.documents,
            service_context=self.service_context,
        )
        query_engine = vector_index.as_query_engine(similarity_top_k=2)
        responses = [query_engine.query(q) for q in test_questions]
        answers = []
        contexts = []
        for r in responses:
            answers.append(r.response)
            contexts.append([c.node.get_content() for c in r.source_nodes])
        dataset_dict = {
            "question": test_questions,
            "answer": answers,
            "contexts": contexts,
        }
        if test_answers is not None:
            dataset_dict["ground_truths"] = [[str(answer)] for answer in test_answers]
        ds = Dataset.from_dict(dataset_dict)
        return ds

    async def aevaluate_models(self, test_questions, test_answers, metrics, response_dataset=None):

        """
        Asynchronously evaluates the generated responses using specified metrics.

        Parameters:
            test_questions (list): The list of test questions.
            test_answers (list): The list of expected answers.
            metrics (list): A list of metrics functions to apply.
            response_dataset (Dataset, optional): A pre-generated dataset to evaluate.

        Returns:
            DataFrame: A pandas DataFrame containing the evaluation results.
        """

        output_file = f"{self.project_path}apps/evals/reports/{self.rag_name}.csv"

        if os.path.exists(output_file):
            print(f"Skipping evaluation as output eval report {output_file} already exists.")
            return pd.read_csv(output_file)

        if not response_dataset:
            response_dataset = self.generate_responses(test_questions, test_answers)
            # print(response_dataset)
        else:
            pass

        result = evaluate(response_dataset,
                          metrics=metrics,
                          is_async=True)
                          # llm=judge_azure_model,
                          # embeddings=judge_azure_embeddings)
        result_df = result.to_pandas()

        # result_df.to_csv(output_file, mode='w', index=False)
        return result_df

    def evaluate_models(self, test_questions, test_answers, metrics, response_dataset=None):
        """
        Synchronously evaluates the generated responses using specified metrics.

        Parameters:
            test_questions (list): The list of test questions.
            test_answers (list): The list of expected answers.
            metrics (list): A list of metrics functions to apply.
            response_dataset (Dataset, optional): A pre-generated dataset to evaluate.

        Returns:
            DataFrame: A pandas DataFrame containing the evaluation results.
        """
        output_file = f"{self.project_path}apps/evals/reports/{self.rag_name}.csv"

        if os.path.exists(output_file):
            print(f"Skipping evaluation as output eval report {output_file} already exists.")
            return pd.read_csv(output_file)

        if not response_dataset:
            response_dataset = self.generate_responses(test_questions, test_answers)
            # print(response_dataset)
        else:
            pass

        result = evaluate(response_dataset,
                          metrics=metrics,
                          llm=judge_azure_model,
                          embeddings=judge_azure_embeddings)
        result_df = result.to_pandas()

        result_df.to_csv(output_file, mode='w', index=False)
        return result_df

    @staticmethod
    def compare_llms(evaluator_1, evaluator_2, test_questions, test_answers, metrics, project_path=None):

        """
        Compares the evaluation results of two language models configured in separate Evaluator instances.

        Parameters:
            evaluator_1 (Evaluator): The first evaluator instance.
            evaluator_2 (Evaluator): The second evaluator instance.
            test_questions (list): A list of test questions to evaluate.
            test_answers (list): A list of correct answers corresponding to the test questions.
            metrics (list): A list of metric functions to use for evaluation.
            project_path (str, optional): Base path for storing the comparison report.

        Returns:
            DataFrame: A pandas DataFrame that shows the comparison between the two sets of results.
        """
        # response_dataset_1 = evaluator_1.generate_responses(test_questions, test_answers)
        # response_dataset_2 = evaluator_2.generate_responses(test_questions, test_answers)
        if evaluator_1.rag_name == evaluator_2.rag_name:
            print("Names of both runs cant be the same. Exiting")
            return
        result_1 = evaluator_1.evaluate_models(test_questions, test_answers, metrics)
        result_2 = evaluator_2.evaluate_models(test_questions, test_answers, metrics)

        print(result_1.head())
        # print(df_2.head())
        # metrics = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall', 'answer_similarity', 'harmfulness']
        metric_string_list = [m.name for m in metrics]

        numeric_values_difference = result_1[metric_string_list] - result_2[metric_string_list]

        # empty_column = pd.DataFrame({'': [''] * result_1.shape[0]})
        result_df = pd.concat([result_1[metric_string_list], result_2[metric_string_list], numeric_values_difference],
                              axis=1)

        # Save the new dataframe to a CSV file
        excel_file = f"{project_path}apps/evals/reports/final_{evaluator_1.rag_name}_{evaluator_2.rag_name}.xlsx"
        # excel_file = f"{project_path}apps/evals/reports/final_.xlsx"
        with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
            result_df.to_excel(writer, sheet_name='Comparison', index=False)

            # Remove gridlines
            workbook = writer.book
            worksheet = writer.sheets['Comparison']
            worksheet.hide_gridlines(2)

            # Define named ranges
            workbook.define_name('RAG_1', f"'Comparison'$Z$1:$Z1")
            workbook.define_name('Result_1',
                                 f"'Comparison'!$A$1:${chr(65 + len(metric_string_list) - 1)}${result_1.shape[0] + 1}")
            workbook.define_name('Result_2',
                                 f"'Comparison'!${chr(65 + len(metric_string_list))}$1:${chr(65 + 2 * len(metric_string_list) - 1)}${result_2.shape[0] + 1}")
            workbook.define_name('Difference',
                                 f"'Comparison'!${chr(65 + 2 * (len(metric_string_list)))}$1:${chr(65 + 3 * len(metric_string_list) - 1)}${numeric_values_difference.shape[0] + 1}")

            # Set values and define named ranges for cells Y1 and Z1
            worksheet.write("$Y$1", evaluator_1.rag_name)
            worksheet.write("$Y$2", evaluator_2.rag_name)
            workbook.define_name('RAG_1', "'Comparison'!$Y$1")
            workbook.define_name('RAG_2', "'Comparison'!$Y$2")
        # result_df.to_csv(f"{self.project_path}apps/evals/reports/metric_eval_report.csv", mode='w',index=False)

        # Usage
        template_file = 'Evaluator Comparison Template.xlsx'
        target_file = f"final_{evaluator_1.rag_name}_{evaluator_2.rag_name}.xlsx"
        copy_sheet(source_file=f"{project_path}apps/evals/reports/{template_file}",
                   target_file=f"{project_path}apps/evals/reports/{target_file}", sheet_name='Results')

        full_file_path = f"{project_path}apps/evals/reports/{target_file}"
        wb = openpyxl.load_workbook(full_file_path)
        # List of named range pairs (source, destination)
        named_ranges = [
            ('Result_1', 'Template_Result_1'),
            ('Result_2', 'Template_Result_2'),
            # ('RAG_1', 'Template_RAG_1'),
            # ('RAG_2', 'Template_RAG_2')

        ]

        # Copy values for each named range pair
        for source_range, destination_range in named_ranges:
            copy_named_range(wb, source_range, destination_range)
        # Save the target workbook
        # Save the workbook

        results_sheet = wb['Results']
        # Create a link to cell Y1 in the 'Comparison' sheet
        if 'Template_RAG_1' in wb.defined_names:
            named_range = wb.defined_names['Template_RAG_1']
            dest = named_range.destinations
            for sheet_name, cell_range in dest:
                wb[sheet_name][cell_range] = '=Comparison!Y1'

        if 'Template_RAG_2' in wb.defined_names:
            named_range = wb.defined_names['Template_RAG_2']
            dest = named_range.destinations
            for sheet_name, cell_range in dest:
                wb[sheet_name][cell_range] = '=Comparison!Y2'

        wb.save(full_file_path)
        return result_df
