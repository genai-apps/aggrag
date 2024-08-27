import shutil
import json, os, sys, asyncio, time, re
from dataclasses import dataclass
from enum import Enum
from typing import List
from statistics import mean, median, stdev
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from pydantic import ValidationError

from library.aggrag.aggrag import AggRAG
from library.aggrag.core.schema import (
    RagStoreBool,
    RagStoreSettings,
    BaseRagSetting,
    SubQARagSetting,
    RaptorRagSetting,
    MetaLlamaRagSetting,
    MetaLangRagSetting,
    TableBaseRagSetting,
)
from library.aggrag.core.utils import zip_directory
from library.providers.dalai import call_dalai
from library.providers import ProviderRegistry
import requests as py_requests
import logging
from werkzeug.utils import secure_filename
from werkzeug.datastructures import ImmutableMultiDict
from library.aggrag.core.log_config import app_loger
from openai import OpenAIError


""" =================
    SETUP AND GLOBALS
    =================
"""
logger = logging.getLogger(__name__)
# Setup Flask app to serve static version of React front-end
HOSTNAME = "localhost"
PORT = 8000
# SESSION_TOKEN = secrets.token_hex(32)
BUILD_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "react-server", "build"
)
STATIC_DIR = os.path.join(BUILD_DIR, "static")

UPLOAD_FILE_MAXIMUM_SIZE: int = 25  # 25 MB

app_loger.configure_logs()


app = Flask(__name__, static_folder=STATIC_DIR, template_folder=BUILD_DIR)
app.config["MAX_CONTENT_LENGTH"] = UPLOAD_FILE_MAXIMUM_SIZE * 1024 * 1024

# Set up CORS for specific routes
cors = CORS(app, resources={r"/*": {"origins": "*"}})

# The cache and examples files base directories
CACHE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "cache")
EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "examples")
CONFIGURATION_DIR = os.path.join(os.getcwd(), "configurations")


class MetricType(Enum):
    KeyValue = 0
    KeyValue_Numeric = 1
    KeyValue_Categorical = 2
    KeyValue_Mixed = 3
    Numeric = 4
    Categorical = 5
    Mixed = 6
    Unknown = 7
    Empty = 8


""" ==============
    UTIL FUNCTIONS
    ==============
"""

HIJACKED_PRINT_LOG_FILE = None
ORIGINAL_PRINT_METHOD = None


def HIJACK_PYTHON_PRINT() -> None:
    # Hijacks Python's print function, so that we can log
    # the outputs when the evaluator is run:
    import builtins
    import tempfile

    global HIJACKED_PRINT_LOG_FILE, ORIGINAL_PRINT_METHOD

    # Create a temporary file for logging and keep it open
    HIJACKED_PRINT_LOG_FILE = tempfile.NamedTemporaryFile(mode="a+", delete=False)

    # Create a wrapper over the original print method, and save the original print
    ORIGINAL_PRINT_METHOD = print

    def hijacked_print(*args, **kwargs):
        if "file" in kwargs:
            # We don't want to override any library that's using print to a file.
            ORIGINAL_PRINT_METHOD(*args, **kwargs)
        else:
            ORIGINAL_PRINT_METHOD(*args, **kwargs, file=HIJACKED_PRINT_LOG_FILE)

    # Replace the original print function with the custom print function
    builtins.print = hijacked_print


def REVERT_PYTHON_PRINT() -> List[str]:
    # Reverts back to original Python print method
    # NOTE: Call this after hijack, and make sure you've caught all exceptions!
    import builtins

    global ORIGINAL_PRINT_METHOD, HIJACKED_PRINT_LOG_FILE

    logs = []
    if HIJACKED_PRINT_LOG_FILE is not None:
        # Read the log file
        HIJACKED_PRINT_LOG_FILE.seek(0)
        logs = HIJACKED_PRINT_LOG_FILE.read().split("\n")

    if ORIGINAL_PRINT_METHOD is not None:
        builtins.print = ORIGINAL_PRINT_METHOD

    HIJACKED_PRINT_LOG_FILE.close()
    HIJACKED_PRINT_LOG_FILE = None

    if len(logs) == 1 and len(logs[0].strip()) == 0:
        logs = []
    return logs


@dataclass
class ResponseInfo:
    """Stores info about a single LLM response. Passed to evaluator functions."""

    text: str  # The text of the LLM response
    prompt: str  # The text of the prompt using to query the LLM
    var: dict  # A dictionary of arguments that filled in the prompt template used to generate the final prompt
    meta: dict  # A dictionary of metadata ('metavars') that is 'carried alongside' data used to generate the prompt
    llm: str  # The name of the LLM queried (the nickname in Aggrag)

    def __str__(self):
        return self.text

    def asMarkdownAST(self):
        import mistune

        md_ast_parser = mistune.create_markdown(renderer="ast")
        return md_ast_parser(self.text)


def check_typeof_vals(arr: list) -> MetricType:
    if len(arr) == 0:
        return MetricType.Empty

    def typeof_set(types: set) -> MetricType:
        if len(types) == 0:
            return MetricType.Empty
        if len(types) == 1 and next(iter(types)) == dict:
            return MetricType.KeyValue
        elif all((t in (int, float) for t in types)):
            # Numeric metrics only
            return MetricType.Numeric
        elif all((t in (str, bool) for t in types)):
            # Categorical metrics only ('bool' is True/False, counts as categorical)
            return MetricType.Categorical
        elif all((t in (int, float, bool, str) for t in types)):
            # Mix of numeric and categorical types
            return MetricType.Mixed
        else:
            # Mix of types beyond basic ones
            return MetricType.Unknown

    def typeof_dict_vals(d):
        dict_val_type = typeof_set(set((type(v) for v in d.values())))
        if dict_val_type == MetricType.Numeric:
            return MetricType.KeyValue_Numeric
        elif dict_val_type == MetricType.Categorical:
            return MetricType.KeyValue_Categorical
        else:
            return MetricType.KeyValue_Mixed

    # Checks type of all values in 'arr' and returns the type
    val_type = typeof_set(set((type(v) for v in arr)))
    if val_type == MetricType.KeyValue:
        # This is a 'KeyValue' pair type. We need to find the more specific type of the values in the dict.
        # First, we check that all dicts have the exact same keys
        for i in range(len(arr) - 1):
            d, e = arr[i], arr[i + 1]
            if set(d.keys()) != set(e.keys()):
                raise Exception(
                    "The keys and size of dicts for evaluation results must be consistent across evaluations."
                )

        # Then, we check the consistency of the type of dict values:
        first_dict_val_type = typeof_dict_vals(arr[0])
        for d in arr[1:]:
            if first_dict_val_type != typeof_dict_vals(d):
                raise Exception(
                    "Types of values in dicts for evaluation results must be consistent across responses."
                )
        # If we're here, all checks passed, and we return the more specific KeyValue type:
        return first_dict_val_type
    else:
        return val_type


def run_over_responses(
    process_func, responses: list, scope: str, process_type: str
) -> list:
    for resp_obj in responses:
        res = resp_obj["responses"]
        if scope == "response":
            # Run process func over every individual response text
            proc = [
                process_func(
                    ResponseInfo(
                        text=r,
                        prompt=resp_obj["prompt"],
                        var=resp_obj["vars"],
                        meta=resp_obj["metavars"] if "metavars" in resp_obj else {},
                        llm=resp_obj["llm"],
                    )
                )
                for r in res
            ]

            if process_type == "processor":
                # Response text was just transformed, not evaluated
                resp_obj["responses"] = proc
            else:
                # Responses were evaluated/scored
                # Check the type of evaluation results
                # NOTE: We assume this is consistent across all evaluations, but it may not be.
                eval_res_type = check_typeof_vals(proc)

                if eval_res_type == MetricType.Numeric:
                    # Store items with summary of mean, median, etc
                    resp_obj["eval_res"] = {
                        "mean": mean(proc),
                        "median": median(proc),
                        "stdev": stdev(proc) if len(proc) > 1 else 0,
                        "range": (min(proc), max(proc)),
                        "items": proc,
                        "dtype": eval_res_type.name,
                    }
                elif eval_res_type in (MetricType.Unknown, MetricType.Empty):
                    raise Exception(
                        "Unsupported types found in evaluation results. Only supported types for metrics are: int, float, bool, str."
                    )
                else:
                    # Categorical, KeyValue, etc, we just store the items:
                    resp_obj["eval_res"] = {
                        "items": proc,
                        "dtype": eval_res_type.name,
                    }
        else:
            # Run process func over the entire response batch
            proc = process_func(
                [
                    ResponseInfo(
                        text=r,
                        prompt=resp_obj["prompt"],
                        var=resp_obj["vars"],
                        llm=resp_obj["llm"],
                    )
                    for r in res
                ]
            )

            if process_type == "processor":
                # Response text was just transformed, not evaluated
                resp_obj["responses"] = proc
            else:
                # Responses were evaluated/scored
                ev_type = check_typeof_vals([proc])
                if ev_type == MetricType.Numeric:
                    resp_obj["eval_res"] = {
                        "mean": proc,
                        "median": proc,
                        "stdev": 0,
                        "range": (proc, proc),
                        "items": [proc],
                        "type": ev_type.name,
                    }
                else:
                    resp_obj["eval_res"] = {
                        "items": [proc],
                        "type": ev_type.name,
                    }
    return responses


async def make_sync_call_async(sync_method, *args, **params):
    """
    Makes a blocking synchronous call asynchronous, so that it can be awaited.
    NOTE: This is necessary for LLM APIs that do not yet support async (e.g. Google PaLM).
    """
    loop = asyncio.get_running_loop()
    method = sync_method
    if len(params) > 0:

        def partial_sync_meth(*a):
            return sync_method(*a, **params)

        method = partial_sync_meth
    return await loop.run_in_executor(None, method, *args)


def exclude_key(d, key_to_exclude):
    return {k: v for k, v in d.items() if k != key_to_exclude}


""" ===================
    FLASK SERVER ROUTES
    ===================
"""


# Serve React app (static; no hot reloading)
@app.route("/")
def index():
    # Get the index.html HTML code
    html_str = render_template("index.html")

    # Inject global JS variables __CF_HOSTNAME and __CF_PORT at the top so that the application knows
    # that it's running from a Flask server, and what the hostname and port of that server is:
    html_str = (
        html_str[:60]
        + f'<script>window.__CF_HOSTNAME="{HOSTNAME}"; window.__CF_PORT={PORT};</script>'
        + html_str[60:]
    )

    return html_str


@app.route("/app/executepy", methods=["POST"])
def executepy():
    """
    Executes a Python function sent from JavaScript,
    over all the `LLMResponse` objects passed in from the front-end.

    POST'd data should be in the form:
    {
        'id': # a unique ID to refer to this information. Used when cache'ing responses.
        'code': str,  # the body of the lambda function to evaluate, in form: lambda responses: <body>
        'responses': List[LLMResponse]  # the responses to run on.
        'scope': 'response' | 'batch'  # the scope of responses to run on --a single response, or all across each batch.
                                       # If batch, evaluator has access to 'responses'. Only matters if n > 1 for each prompt.
        'process_type': 'evaluator' | 'processor'  # the type of processing to perform. Evaluators only 'score'/annotate responses. Processors change responses (e.g. text).
        'script_paths': unspecified | List[str]  # the paths to scripts to be added to the path before the lambda function is evaluated
    }

    NOTE: This should only be run on your server on code you trust.
          There is no sandboxing; no safety. We assume you are the creator of the code.
    """
    data = request.get_json()

    # Check that all required info is here:
    if not set(data.keys()).issuperset({"id", "code", "responses", "scope"}):
        return jsonify({"error": "POST data is improper format."})
    if not isinstance(data["id"], str) or len(data["id"]) == 0:
        return jsonify(
            {"error": "POST data id is improper format (length 0 or not a string)."}
        )
    if data["scope"] not in ("response", "batch"):
        return jsonify(
            {
                "error": "POST data scope is unknown. Must be either 'response' or 'batch'."
            }
        )

    # Check format of responses:
    responses = data["responses"]
    if (isinstance(responses, str) or not isinstance(responses, list)) or (
        len(responses) > 0 and any([not isinstance(r, dict) for r in responses])
    ):
        return jsonify({"error": "POST data responses is improper format."})

    # Add the path to any scripts to the path:
    try:
        if "script_paths" in data:
            for script_path in data["script_paths"]:
                # get the folder of the script_path:
                script_folder = os.path.dirname(script_path)
                # check that the script_folder is valid, and it contains __init__.py
                if not os.path.exists(script_folder):
                    print(script_folder, "is not a valid script path.")
                    continue

                # add it to the path:
                sys.path.append(script_folder)
                print(f"added {script_folder} to sys.path")
    except Exception as e:
        return jsonify(
            {
                "error": f"Could not add script path to sys.path. Error message:\n{str(e)}"
            }
        )

    # Get processor type, if any
    process_type = data["process_type"] if "process_type" in data else "evaluator"

    # Create the evaluator function
    # DANGER DANGER!
    try:
        exec(data["code"], globals())

        # Double-check that there is an 'evaluate' method in our namespace.
        # This will throw a NameError if not:
        if process_type == "evaluator":
            evaluate  # noqa
        else:
            process  # noqa
    except Exception as e:
        return jsonify(
            {"error": f"Could not compile evaluator code. Error message:\n{str(e)}"}
        )

    evald_responses = []
    logs = []
    try:
        HIJACK_PYTHON_PRINT()
        evald_responses = run_over_responses(
            evaluate if process_type == "evaluator" else process,
            responses,
            scope=data["scope"],
            process_type=process_type,
        )  # noqa
        logs = REVERT_PYTHON_PRINT()
    except Exception as e:
        logs = REVERT_PYTHON_PRINT()
        return jsonify(
            {
                "error": f'Error encountered while trying to run "evaluate" method:\n{str(e)}',
                "logs": logs,
            }
        )

    ret = jsonify({"responses": evald_responses, "logs": logs})
    ret.headers.add("Access-Control-Allow-Origin", "*")
    return ret


@app.route("/app/fetchExampleFlow", methods=["POST"])
def fetchExampleFlow():
    """
    Fetches the example flow data, given its filename. The filename should be the
    name of a file in the examples/ folder of the package.

    Used for loading examples in the Example Flow modal.

    POST'd data should be in form:
    {
        name: <str>  # The filename (without .cforge extension)
    }
    """
    # Verify post'd data
    data = request.get_json()
    if "name" not in data:
        return jsonify({"error": 'Missing "name" parameter to fetchExampleFlow.'})

    # Verify 'examples' directory exists:
    if not os.path.isdir(EXAMPLES_DIR):
        dirpath = os.path.dirname(os.path.realpath(__file__))
        return jsonify(
            {"error": f"Could not find an examples/ directory at path {dirpath}"}
        )

    # Check if the file is there:
    filepath = os.path.join(EXAMPLES_DIR, data["name"] + ".cforge")
    if not os.path.isfile(filepath):
        return jsonify(
            {"error": f"Could not find an example flow named {data['name']}"}
        )

    # Load the file and return its data:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            filedata = json.load(f)
    except Exception as e:
        return jsonify({"error": f"Error parsing example flow at {filepath}: {str(e)}"})

    ret = jsonify({"data": filedata})
    ret.headers.add("Access-Control-Allow-Origin", "*")
    return ret


@app.route("/app/fetchOpenAIEval", methods=["POST"])
def fetchOpenAIEval():
    """
    Fetches a preconverted OpenAI eval as a .cforge JSON file.

    First detects if the eval is already in the cache. If the eval is already downloaded,
    it will be stored in examples/ folder of the package under a new oaievals directory.
    If it's not in the cache, it will download it from the webserver.

    POST'd data should be in form:
    {
        'name': <str>  # The name of the eval to grab (without .cforge extension)
    }
    """
    # Verify post'd data
    data = request.get_json()
    if "name" not in data:
        return jsonify({"error": 'Missing "name" parameter to fetchOpenAIEval.'})
    evalname = data["name"]

    # Verify 'examples' directory exists:
    if not os.path.isdir(EXAMPLES_DIR):
        dirpath = os.path.dirname(os.path.realpath(__file__))
        return jsonify(
            {"error": f"Could not find an examples/ directory at path {dirpath}"}
        )

    # Check if an oaievals subdirectory exists; if so, check for the file; if not create it:
    oaievals_cache_dir = os.path.join(EXAMPLES_DIR, "oaievals")
    if os.path.isdir(oaievals_cache_dir):
        filepath = os.path.join(oaievals_cache_dir, evalname + ".cforge")
        if os.path.isfile(filepath):
            # File was already downloaded. Load it from cache:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    filedata = json.load(f)
            except Exception as e:
                return jsonify(
                    {
                        "error": f"Error parsing OpenAI evals flow at {filepath}: {str(e)}"
                    }
                )
            ret = jsonify({"data": filedata})
            ret.headers.add("Access-Control-Allow-Origin", "*")
            return ret
        # File was not downloaded
    else:
        # Directory does not exist yet; create it
        try:
            os.mkdir(oaievals_cache_dir)
        except Exception as e:
            return jsonify(
                {
                    "error": f"Error creating a new directory 'oaievals' at filepath {oaievals_cache_dir}: {str(e)}"
                }
            )

    # Download the preconverted OpenAI eval from the GitHub main branch for aggrag
    _url = f"https://github.com/genai-apps/aggrag"
    response = py_requests.get(_url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the response as JSON
        filedata = response.json()

        # Store to the cache:
        with open(
            os.path.join(oaievals_cache_dir, evalname + ".cforge"), "w", encoding="utf8"
        ) as f:
            json.dump(filedata, f)
    else:
        print("Error:", response.status_code)
        return jsonify(
            {
                "error": f"Error downloading OpenAI evals flow from {_url}: status code {response.status_code}"
            }
        )

    ret = jsonify({"data": filedata})
    ret.headers.add("Access-Control-Allow-Origin", "*")
    return ret


@app.route("/app/fetchEnvironAPIKeys", methods=["POST"])
def fetchEnvironAPIKeys():
    keymap = {
        "OPENAI_API_KEY": "OpenAI",
        "OPENAI_BASE_URL": "OpenAI_BaseURL",
        "ANTHROPIC_API_KEY": "Anthropic",
        "PALM_API_KEY": "Google",
        "HUGGINGFACE_API_KEY": "HuggingFace",
        "AZURE_OPENAI_KEY": "Azure_OpenAI",
        "AZURE_OPENAI_ENDPOINT": "Azure_OpenAI_Endpoint",
        "ALEPH_ALPHA_API_KEY": "AlephAlpha",
        "AWS_ACCESS_KEY_ID": "AWS_Access_Key_ID",
        "AWS_SECRET_ACCESS_KEY": "AWS_Secret_Access_Key",
        "AWS_REGION": "AWS_Region",
        "AWS_SESSION_TOKEN": "AWS_Session_Token",
        "TOGETHER_API_KEY": "Together",
    }
    d = {alias: os.environ.get(key) for key, alias in keymap.items()}
    ret = jsonify(d)
    ret.headers.add("Access-Control-Allow-Origin", "*")
    return ret


@app.route("/app/makeFetchCall", methods=["POST"])
def makeFetchCall():
    """
    Use in place of JavaScript's 'fetch' (with POST method), in cases where
    cross-origin policy blocks responses from client-side fetches.

    POST'd data should be in form:
    {
        'url': <str>  # the url to fetch from
        'headers': <dict>  # a JSON object of the headers
        'body': <dict>  # the request payload, as JSON
    }
    """
    # Verify post'd data
    data = request.get_json()
    if not set(data.keys()).issuperset({"url", "headers", "body"}):
        return jsonify({"error": "POST data is improper format."})

    url = data["url"]
    headers = data["headers"]
    body = data["body"]

    response = py_requests.post(url, headers=headers, json=body)

    if response.status_code == 200:
        ret = jsonify({"response": response.json()})
        ret.headers.add("Access-Control-Allow-Origin", "*")
        return ret
    else:
        err_msg = "API request to Anthropic failed"
        ret = response.json()
        if "error" in ret and "message" in ret["error"]:
            err_msg += ": " + ret["error"]["message"]
        return jsonify({"error": err_msg})


@app.route("/app/callDalai", methods=["POST"])
async def callDalai():
    """
    Fetch response from a Dalai-hosted model (Alpaca or Llama).
    Requires Python backend since depends on custom library code to extract response.

    POST'd data should be a dict of keyword arguments to provide the call_dalai method.
    """
    # Verify post'd data
    data = request.get_json()
    if not set(data.keys()).issuperset(
        {"prompt", "model", "server", "n", "temperature"}
    ):
        return jsonify({"error": "POST data is improper format."})

    try:
        query, response = await call_dalai(**data)
    except Exception as e:
        return jsonify({"error": str(e)})

    ret = jsonify({"query": query, "response": response})
    ret.headers.add("Access-Control-Allow-Origin", "*")
    return ret


@app.route("/app/initCustomProvider", methods=["POST"])
def initCustomProvider():
    """
    Initalizes custom model provider(s) defined in a Python script,
    and returns specs for the front-end UI provider dropdown and the providers' settings window.

    POST'd data should be in form:
    {
        'code': <str>  # the Python script to save + execute,
    }
    """
    # Verify post'd data
    data = request.get_json()
    if "code" not in data:
        return jsonify({"error": "POST data is improper format."})

    # Sanity check that the code actually registers a provider
    if "@provider" not in data["code"]:
        return jsonify(
            {
                "error": """Did not detect a @provider decorator. Custom provider scripts should register at least one @provider. 
                                    Do `from library.providers import provider` and decorate your provider completion function with @provider."""
            }
        )

    # Establish the custom provider script cache directory
    provider_scripts_dir = os.path.join(CACHE_DIR, "provider_scripts")
    if not os.path.isdir(provider_scripts_dir):
        # Create the directory
        try:
            os.makedirs(provider_scripts_dir, exist_ok=True)
        except Exception as e:
            return jsonify(
                {
                    "error": f"Error creating a new directory 'provider_scripts' at filepath {provider_scripts_dir}: {str(e)}"
                }
            )

    # For keeping track of what script registered providers came from
    script_id = str(round(time.time() * 1000))
    ProviderRegistry.set_curr_script_id(script_id)
    ProviderRegistry.watch_next_registered()

    # Attempt to run the Python script, in context
    try:
        exec(data["code"], globals(), None)

        # This should have registered one or more new CustomModelProviders.
    except Exception as e:
        return jsonify(
            {"error": f"Error while executing custom provider code:\n{str(e)}"}
        )

    # Check whether anything was updated, and what
    new_registries = ProviderRegistry.last_registered()
    if (
        len(new_registries) == 0
    ):  # Determine whether there's at least one custom provider.
        return jsonify(
            {
                "error": "Did not detect any custom providers added to the registry. Make sure you are registering your provider with @provider correctly."
            }
        )

    # At least one provider was registered; detect if it had a past script id and remove those file(s) from the cache
    if any((v is not None for v in new_registries.values())):
        # For every registered provider that was overwritten, remove the cache'd script(s) associated with it:
        past_script_ids = [v for v in new_registries.values() if v is not None]
        for sid in past_script_ids:
            past_script_path = os.path.join(provider_scripts_dir, f"{sid}.py")
            try:
                if os.path.isfile(past_script_path):
                    os.remove(past_script_path)
            except Exception as e:
                return jsonify(
                    {
                        "error": f"Error removing cache'd custom provider script at filepath {past_script_path}: {str(e)}"
                    }
                )

    # Get the names and specs of all currently registered CustomModelProviders,
    # and pass that info to the front-end (excluding the func):
    registered_providers = [exclude_key(d, "func") for d in ProviderRegistry.get_all()]

    # Copy the passed Python script to a local file in the package directory
    try:
        with open(os.path.join(provider_scripts_dir, f"{script_id}.py"), "w") as f:
            f.write(data["code"])
    except Exception as e:
        return jsonify(
            {
                "error": f"Error saving script 'provider_scripts' at filepath {provider_scripts_dir}: {str(e)}"
            }
        )

    # Return all loaded providers
    return jsonify({"providers": registered_providers})


@app.route("/app/loadCachedCustomProviders", methods=["POST"])
def loadCachedCustomProviders():
    """
    Initalizes all custom model provider(s) in the local provider_scripts directory.
    """
    provider_scripts_dir = os.path.join(CACHE_DIR, "provider_scripts")
    if not os.path.isdir(provider_scripts_dir):
        # No providers to load.
        return jsonify({"providers": []})

    try:
        for file_name in os.listdir(provider_scripts_dir):
            file_path = os.path.join(provider_scripts_dir, file_name)
            if os.path.isfile(file_path) and os.path.splitext(file_path)[1] == ".py":
                # For keeping track of what script registered providers came from
                ProviderRegistry.set_curr_script_id(os.path.splitext(file_name)[0])

                # Read the Python script
                with open(file_path, "r") as f:
                    code = f.read()

                # Try to execute it in the global context
                try:
                    exec(code, globals(), None)
                except Exception as code_exc:
                    # Remove the script file associated w the failed execution
                    os.remove(file_path)
                    raise code_exc
    except Exception as e:
        return jsonify(
            {"error": f"Error while loading custom providers from cache: \n{str(e)}"}
        )

    # Get the names and specs of all currently registered CustomModelProviders,
    # and pass that info to the front-end (excluding the func):
    registered_providers = [exclude_key(d, "func") for d in ProviderRegistry.get_all()]

    return jsonify({"providers": registered_providers})


@app.route("/app/removeCustomProvider", methods=["POST"])
def removeCustomProvider():
    """
    Initalizes custom model provider(s) defined in a Python script,
    and returns specs for the front-end UI provider dropdown and the providers' settings window.

    POST'd data should be in form:
    {
        'name': <str>  # a name that refers to the registered custom provider in the `ProviderRegistry`
    }
    """
    # Verify post'd data
    data = request.get_json()
    name = data.get("name")
    if name is None:
        return jsonify({"error": "POST data is improper format."})

    if not ProviderRegistry.has(name):
        return jsonify({"error": f'Could not find a custom provider named "{name}"'})

    # Get the script id associated with the provider we're about to remove
    script_id = ProviderRegistry.get(name).get("script_id")

    # Remove the custom provider from the registry
    ProviderRegistry.remove(name)

    # Attempt to delete associated script from cache
    if script_id:
        script_path = os.path.join(CACHE_DIR, "provider_scripts", f"{script_id}.py")
        if os.path.isfile(script_path):
            os.remove(script_path)

    return jsonify({"success": True})


@app.route("/app/callCustomProvider", methods=["POST"])
async def callCustomProvider():
    """
    Calls a custom model provider and returns the response.

    POST'd data should be in form:
    {
        'name': <str>  # the name of the provider in the `ProviderRegistry`
        'params': <dict>  # the params (prompt, model, etc) to pass to the provider function.
    }
    """
    # Verify post'd data
    data = request.get_json()
    if not set(data.keys()).issuperset({"name", "params"}):
        return jsonify({"error": "POST data is improper format."})

    # Load the name of the provider
    name = data["name"]
    params = data["params"]

    # Double-check that the custom provider exists in the registry, and (if passed) a model with that name exists
    provider_spec = ProviderRegistry.get(name)
    if provider_spec is None:
        return jsonify(
            {
                "error": f"Could not find provider named {name}. Perhaps you need to import a custom provider script?"
            }
        )

    # Call + await the custom provider function, passing in the JSON payload as kwargs
    try:
        response = await make_sync_call_async(provider_spec.get("func"), **params)
    except Exception as e:
        return jsonify(
            {
                "error": f"Error encountered while calling custom provider function: {str(e)}"
            }
        )

    # Return the response
    print(f"response from {ProviderRegistry.get(name)}: {response} ")
    return jsonify({"response": response})


@app.route("/app/uploadFile", methods=["POST"])
async def uploadFileToKnowledgeBase():
    """
    Uploads pdf/word file, given its filename and file_node_id.
    POST'd data should be in form:
    {
        timestamp: <str>  # The timestamp (last 7 digits),
        file_node_id: <str> # The UUID for file node,
        file: <fileobj>,
        p_folder: <str> # The parent folder (usecase)
        i_folder: <str> # The iteration folder
    }
    """
    ALLOWED_EXTENSIONS = {"txt", "pdf", "doc", "docx"}

    def allowed_file(filename):
        return (
            "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
        )

    # Verify post'd data
    if "file" not in request.files:
        return jsonify({"error": "Missing file parameter to upload file."})

    # if request.files["file"] > UPLOAD_FILE_MAXIMUM_SIZE * 1024 * 1024:
    #     return jsonify(
    #         {"error": "File exceeds maximum size limit set by the administrator."}
    #     )

    req = request.form
    if "p_folder" not in req or "i_folder" not in req:
        return jsonify({"error": "Missing usecase or iteration to upload file."})
    if "file_node_id" not in req:
        return jsonify({"error": "Missing file_node_id parameter to upload file."})
    if "timestamp" not in req:
        return jsonify({"error": "Missing timestamp parameter to upload file."})

    # Verify 'raw_docs' directory exists:
    RAW_DOCS_DIR = os.path.join(
        CONFIGURATION_DIR,
        req["p_folder"],
        req["i_folder"],
        "raw_docs",
        req["file_node_id"] + "_" + req["timestamp"],
    )
    print(RAW_DOCS_DIR)
    if not os.path.isdir(RAW_DOCS_DIR):
        os.makedirs(RAW_DOCS_DIR)

    # Save the file and return:
    try:
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Missing filename parameter to upload file."})
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print("Filename data: ", filename)
            file.save(os.path.join(RAW_DOCS_DIR, filename))
            return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": f"Error saving file: {str(e)}"})


@app.route("/app/indexRAGFiles", methods=["POST"])
def indexRAGFiles():
    """
    Indexes files based on the provided file path and RAG name.
    API accepts a JSON request body containing the following parameters:
    `files_path` (str): The path to the directory containing the files to be indexed.
    `rag_name` (str): The name of the RAG model to be used for indexing.

    Parameters:
        request (Request): A JSON body with `files_path` and `rag_name`.

    Returns:
        A response object indicating the success or failure of the indexing operation

    Example JSON request body:
    {
        "files_path": "/path/to/files",
        "rag_name": "example_rag"
    }

    Example JSON response:
    {
        "status": "success"
    }

    """
    data = request.get_json()
    working_dir = data.get("files_path")
    rag_name = data.get("rag_name")
    use_case_name = working_dir.split("/")[1]
    iteration = working_dir.split("/")[2]

    raw_docs_path = os.path.dirname(working_dir)

    rag_models = ["base", "raptor", "subqa", "meta_llama", "meta_lang", "tableBase"]
    if rag_name not in rag_models:
        return jsonify(
            {
                "error": f"Invalid rag_name, rag_name can be one of this values {rag_models}"
            }
        )

    ragstore_settings = data.get("ragstore_settings")

    if not ragstore_settings:
        return jsonify({"error": f"ragstore_settings is required"})

    allowed_rag_settings = {
        "base_rag_setting",
        "subqa_rag_setting",
        "raptor_rag_setting",
        "meta_llama_rag_setting",
        "meta_lang_rag_setting",
        "tableBase_rag_setting",
    }
    if not set(ragstore_settings.keys()).issubset(allowed_rag_settings):
        return jsonify(
            {
                "error": 'ragstore_settings should be one of ["base_rag_setting", "subqa_rag_setting", "raptor_rag_setting", "meta_llama_rag_setting", "meta_lang_rag_setting", "tableBase_rag_setting"]'
            }
        )

    try:
        aggrag_ragstore_settings = RagStoreSettings(**ragstore_settings)
        ragstore_settings = RagStoreSettings(
            base_rag_setting=(
                BaseRagSetting(**ragstore_settings.get("base_rag_setting"))
                if ragstore_settings.get("base_rag_setting")
                else None
            ),
            # base_rag_setting=BaseRagSetting(**ragstore_settings.base_rag_setting.model_dump()) if ragstore_settings.base_rag_setting else None,
            raptor_rag_setting=(
                RaptorRagSetting(**ragstore_settings.get("raptor_rag_setting"))
                if ragstore_settings.get("raptor_rag_setting")
                else None
            ),
            subqa_rag_setting=(
                SubQARagSetting(**ragstore_settings.get("subqa_rag_setting"))
                if ragstore_settings.get("subqa_rag_setting")
                else None
            ),
            meta_llama_rag_setting=(
                MetaLlamaRagSetting(**ragstore_settings.get("meta_llama_rag_setting"))
                if ragstore_settings.get("meta_llama_rag_setting")
                else None
            ),
            meta_lang_rag_setting=(
                MetaLangRagSetting(**ragstore_settings.get("meta_lang_rag_setting"))
                if ragstore_settings.get("meta_lang_rag_setting")
                else None
            ),
            tableBase_rag_setting=(
                TableBaseRagSetting(**ragstore_settings.get("tableBase_rag_setting"))
                if ragstore_settings.get("tableBase_rag_setting")
                else None
            ),
        )

    except ValidationError as e:
        # Return a JSON response with the validation errors
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        # Handle other exceptions
        return jsonify({'error': str(e)}), 500




    aggrag = AggRAG(
        ragstore_bool=RagStoreBool(**{rag_name: True}),
        usecase_name=use_case_name,
        iteration=iteration,
        DATA_DIR=working_dir,
        ragstore_settings=aggrag_ragstore_settings,
    )

    try:
        aggrag.documents_loader(DIR=raw_docs_path)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)})

    loop = get_event_loop()
    index_task = loop.run_until_complete(
        aggrag.create_all_index_async(documents=aggrag.documents)
    )

    if not index_task:
        logger.info("Issue in index formation extraction")
        return jsonify({"error": f"Issue in index formation extraction"})

    return jsonify({"response": "success"})


@app.route("/app/createiteration", methods=["POST"])
def create_iter_folder():
    data = request.get_json()
    usecase_folder_name = data.get("folderName")

    parent_dir = os.path.join(os.getcwd(), "configurations")
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    folder_path = os.path.join(parent_dir, usecase_folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Fetch existing iterations
    existing_iterations = [
        f
        for f in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, f)) and f.startswith("iteration ")
    ]
    iteration_numbers = []
    for f in existing_iterations:
        match = re.match(r"iteration (\d+)", f)
        if match:
            iteration_numbers.append(int(match.group(1)))

    next_iteration_num = max(iteration_numbers) + 1 if iteration_numbers else 1
    iter_folder_name = f"iteration {next_iteration_num}"
    iteration_folder_path = os.path.join(folder_path, iter_folder_name)

    if not os.path.exists(iteration_folder_path):
        os.makedirs(iteration_folder_path)
        return (
            jsonify(
                {
                    "message": "Folder created successfully!",
                    "iter_folder_name": iter_folder_name,
                }
            ),
            201,
        )
    else:
        return jsonify({"message": "Folder already exists."}), 400


def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def read_and_modify_json(file_path):
    try:
        with open(file_path, "r") as f:
            file_content = json.load(f)
        if "flow" in file_content and "isCommitted" in file_content["flow"]:
            file_content["flow"]["isCommitted"] = False
        return file_content
    except Exception as e:
        raise Exception(f"Failed to read or modify JSON: {str(e)}")


def write_json(file_content, file_path):
    try:
        with open(file_path, "w") as f:
            json.dump(file_content, f, indent=4)
    except Exception as e:
        raise Exception(f"Failed to write JSON: {str(e)}")


def generate_unique_folder_name(base_folder_name, parent_path):
    existing_iterations = [
        f
        for f in os.listdir(parent_path)
        if os.path.isdir(os.path.join(parent_path, f)) and f.startswith("iteration ")
    ]
    iteration_numbers = []
    for f in existing_iterations:
        match = re.match(r"iteration (\d+)", f)
        if match:
            iteration_numbers.append(int(match.group(1)))

    next_iteration_num = max(iteration_numbers) + 1 if iteration_numbers else 1
    new_iter_folder_name = f"iteration {next_iteration_num}"

    return new_iter_folder_name


@app.route("/app/copyiteration", methods=["POST"])
def copy_iteration():
    data = request.get_json()
    usecase_folder_name = data.get("folderName")
    iter_folder_name = data.get("iterationName")
    file_name = data.get("fileName")

    parent_dir = os.path.join(os.getcwd(), "configurations")
    ensure_directory_exists(parent_dir)

    folder_path = os.path.join(parent_dir, usecase_folder_name)
    ensure_directory_exists(folder_path)

    new_iter_folder_name = generate_unique_folder_name(iter_folder_name, folder_path)
    iteration_folder_path = os.path.join(folder_path, new_iter_folder_name)

    ensure_directory_exists(iteration_folder_path)

    source_file_path = os.path.join(folder_path, iter_folder_name, file_name)

    if os.path.isfile(source_file_path):
        try:
            file_content = read_and_modify_json(source_file_path)
            new_file_path = os.path.join(iteration_folder_path, file_name)
            write_json(file_content, new_file_path)
            return (
                jsonify(
                    {
                        "message": "Folder created and file copied successfully!",
                        "iter_folder_name": new_iter_folder_name,
                        "file_name": file_name,
                    }
                ),
                201,
            )
        except Exception as e:
            return jsonify({"message": f"Failed to copy file: {str(e)}"}), 500
    else:
        return (
            jsonify(
                {
                    "message": "Folder created successfully, but no valid file provided!",
                    "iter_folder_name": new_iter_folder_name,
                }
            ),
            201,
        )


def generate_unique_usecase_folder_name(base_folder_name, parent_path, id):
    base_folder = base_folder_name.split("__")
    new_folder_name = base_folder[0] + "_Copy" + "__" + id
    counter = 1
    while os.path.exists(os.path.join(parent_path, new_folder_name)):
        new_folder_name = f"{base_folder[0]}_Copy{counter}__{id}"
        counter += 1
    return new_folder_name


@app.route("/app/copyusecase", methods=["POST"])
def copy_usecase():
    data = request.get_json()
    source_usecase_folder_name = data.get("sourceUsecase")
    user_provided_target_usecase_folder = data.get("targetUsecase")
    aggrag_user_id = data.get("aggrag_user_id")
    parent_dir = os.path.join(os.getcwd(), "configurations")

    ensure_directory_exists(parent_dir)

    source_usecase_folder_path = os.path.join(parent_dir, source_usecase_folder_name)
    if not os.path.exists(source_usecase_folder_path):
        return jsonify({"message": "Source use case folder does not exist"}), 404
    isUsecaseAlreadyExists = False
    for usecase in os.listdir(parent_dir):
        if user_provided_target_usecase_folder in usecase:
            isUsecaseAlreadyExists = True
            break
        else:
            isUsecaseAlreadyExists = False
    if isUsecaseAlreadyExists:
        # target_usecase_folder_name = generate_unique_usecase_folder_name(user_provided_target_usecase_folder,parent_dir,aggrag_user_id)
        return jsonify({"ok": False, "message": "Usecase folder already exists"}), 400
    else:
        target_usecase_folder_name = user_provided_target_usecase_folder
    # target_usecase_folder_name = user_provided_target_usecase_folder
    target_usecase_folder_path = os.path.join(parent_dir, target_usecase_folder_name)
    ensure_directory_exists(target_usecase_folder_path)

    for item in os.listdir(source_usecase_folder_path):
        item_path = os.path.join(source_usecase_folder_path, item)
        if os.path.isdir(item_path):
            target_iteration_folder_path = os.path.join(
                target_usecase_folder_path, item
            )
            ensure_directory_exists(target_iteration_folder_path)
            for file in os.listdir(item_path):
                if file.endswith(".cforge"):
                    source_file_path = os.path.join(item_path, file)
                    target_file_path = os.path.join(target_iteration_folder_path, file)
                    try:
                        file_content = read_and_modify_json(source_file_path)
                        metadata_path = os.path.join(
                            target_usecase_folder_path, ".metadata"
                        )
                        with open(metadata_path, "w") as f:
                            json.dump(
                                {"created_at": time.time(), "user_id": aggrag_user_id},
                                f,
                            )
                        write_json(file_content, target_file_path)
                    except Exception as e:
                        return (
                            jsonify({"message": f"Failed to copy file: {str(e)}"}),
                            500,
                        )
        else:
            target_file_path = os.path.join(target_usecase_folder_path, item)
            try:
                file_content = read_and_modify_json(item_path)
                write_json(file_content, target_file_path)
            except Exception as e:
                return jsonify({"message": f"Failed to copy file: {str(e)}"}), 500

    iterations_info = []

    for item in os.listdir(target_usecase_folder_path):
        item_path = os.path.join(target_usecase_folder_path, item)
        # Check if the item is a directory (i.e., an iteration folder)
        if os.path.isdir(item_path):
            # Ensure that the target iteration folder exists
            target_iteration_folder_path = os.path.join(
                target_usecase_folder_path, item
            )
            if os.path.exists(target_iteration_folder_path):
                iteration_files = []

                for file in os.listdir(item_path):
                    if file.endswith(".cforge"):
                        # Just collect the file names,
                        iteration_files.append(file)

                iterations_info.append(
                    {"iteration_name": item, "files": iteration_files}
                )

    return (
        jsonify(
            {
                "message": "Use case and iterations copied successfully!",
                "target_usecase_folder_name": target_usecase_folder_name,
                "iterations_info": iterations_info,
            }
        ),
        201,
    )


@app.route("/app/createusecase", methods=["POST"])
def create_usecase():
    data = request.get_json()
    folder_name = data.get("folderName")
    user_id = data.get("aggrag_user_id")
    parent_dir = os.path.join(os.getcwd(), "configurations")

    if "__default" in folder_name or "__Default" in folder_name:
        # folder_with_userid = folder_name
        return jsonify({"ok": False, "message": "Use case name not allowed"}), 400
    else:
        folder_with_userid = folder_name + "__" + user_id

    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    folder_path = os.path.join(parent_dir, folder_with_userid)

    if os.path.exists(folder_path):
        return jsonify({"ok": False, "message": "Usecase folder already exists"}), 400

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        # Create a metadata file with the creation time
        metadata_path = os.path.join(folder_path, ".metadata")
        with open(metadata_path, "w") as f:
            json.dump({"created_at": time.time(), "user_id": user_id}, f)
    iteration_folder_path = os.path.join(folder_path, "iteration 1")
    os.makedirs(iteration_folder_path)
    return (
        jsonify(
            {
                "ok": True,
                "message": "Folder created successfully!",
                "usecase_folder": folder_with_userid,
            }
        ),
        200,
    )


def read_cforge_file(file_path):
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
            # Access the nested isCommitted value
            return data.get("flow", {}).get("isCommitted", False)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return False


@app.route("/app/loadcforge", methods=["GET"])
def load_cforge():
    aggrag_user_id = request.args.get("aggrag_user_id")
    parent_dir = os.path.join(os.getcwd(), "configurations")
    if not os.path.exists(parent_dir):
        return jsonify([])
    local_key = os.getenv("REACT_APP_API_URL")
    # we can use this local key to decide whether the user is in local or in server

    # Fetch folders and read .metadata files to get creation time and user_id
    folder_data_list = []

    for folder in os.listdir(parent_dir):
        folder_path = os.path.join(parent_dir, folder)
        metadata_path = os.path.join(folder_path, ".metadata")

        if os.path.isdir(folder_path) and os.path.exists(metadata_path):
            with open(metadata_path, "r") as metadata_file:
                metadata = json.load(metadata_file)
                if "localhost" in local_key:

                    folder_data_list.append(
                        {
                            "label": folder,
                            "creation_time": metadata.get(
                                "created_at", 0
                            ),  # Use 0 if 'created_at' is missing
                            "path": folder_path,
                            "default": "__default" in folder or "__Default" in folder,
                        }
                    )
                elif (
                    metadata.get("user_id") == aggrag_user_id
                    or "__default" in folder
                    or "__Default" in folder
                ):

                    # Append folder data with creation time
                    folder_data_list.append(
                        {
                            "label": folder,
                            "creation_time": metadata.get(
                                "created_at", 0
                            ),  # Use 0 if 'created_at' is missing
                            "path": folder_path,
                            "default": "__default" in folder or "__Default" in folder,
                        }
                    )
    # Sort folders based on creation time
    folder_data_list.sort(key=lambda x: (not x["default"], x["creation_time"]))

    structured_folders = []

    for folder_data in folder_data_list:
        folder_path = folder_data["path"]
        folder_contents = os.listdir(folder_path)

        folder_data_structured = {"label": folder_data["label"], "subItems": []}

        # Fetch and sort iterations numerically
        iterations = sorted(
            [
                item
                for item in folder_contents
                if os.path.isdir(os.path.join(folder_path, item))
                and re.match(r"iteration[ _](\d+)", item)
            ],
            key=lambda x: int(re.match(r"iteration[ _](\d+)", x).group(1)),
        )

        for iteration in iterations:
            iteration_path = os.path.join(folder_path, iteration)
            iteration_contents = os.listdir(iteration_path)
            iteration_data = []

            for sub_item in iteration_contents:
                sub_item_path = os.path.join(iteration_path, sub_item)
                if sub_item.endswith(".cforge"):
                    is_committed = read_cforge_file(sub_item_path)
                    iteration_data.append(
                        {"label": sub_item, "isCommitted": is_committed}
                    )

            folder_data_structured["subItems"].append(
                {"label": iteration, "subItems": iteration_data}
            )

        structured_folders.append(folder_data_structured)

    return jsonify(structured_folders if structured_folders else [])


@app.route("/app/saveflow", methods=["POST"])
def save_flow():
    data = request.get_json()

    # Retrieve flow and cache data from request JSON
    flow_data = data.get("flow")
    cache_data = data.get("cache")
    folder_name = data.get("folderName")
    iteration_name = data.get("iterationName")
    file_name = data.get("fileName")  # This can be an empty string

    # Check if folder name or iteration name is empty
    if not folder_name or not iteration_name:
        return jsonify({"message": "File not saved as path is not mentioned"}), 400

    # Ensure 'configurations' directory exists in the current working directory
    parent_dir = os.path.join(os.getcwd(), "configurations")
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    # Create use case folder (e.g., 'usecase1')
    use_case_dir = os.path.join(parent_dir, folder_name)
    if not os.path.exists(use_case_dir):
        os.makedirs(use_case_dir)

    iteration_dir = os.path.join(use_case_dir, iteration_name)
    if not os.path.exists(iteration_dir):
        os.makedirs(iteration_dir)

    # Use provided file name or create a new one with the timestamp
    if not file_name:
        file_name = f"flow-{data.get('timestamp')}.cforge"

    file_path = os.path.join(iteration_dir, file_name)

    try:
        flow_and_cache = {"flow": flow_data, "cache": cache_data}

        # Save the file (overwrite if it exists)
        with open(file_path, "w") as file:
            json.dump(flow_and_cache, file, indent=4)

        return (
            jsonify(
                {
                    "message": "Flow saved successfully!",
                    "file_path": file_path,
                    "file_name": file_name,
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"message": f"Failed to save flow: {str(e)}"}), 500


@app.route("/app/getfile", methods=["GET"])
def get_file():
    file_path = request.args.get("file_path")

    base_dir = os.path.join(os.getcwd(), "configurations")
    full_path = os.path.join(base_dir, file_path)

    if not full_path.startswith(base_dir):
        return jsonify({"message": "Invalid file path."}), 400

    if os.path.exists(full_path) and os.path.isfile(full_path):
        return send_file(full_path)
    else:
        return jsonify({"message": "File not found."}), 404


@app.route("/app/RAGStoreChat", methods=["POST"])
def RAGStoreChat():
    """
    Handles a chat request by using the specified index path and RAG model.
    API accepts a JSON request body containing the following parameters:
    `index_path` (str): The path to the directory containing the indexed files.
    `rag_name` (str): The name of the RAG model to be used for generating the response.
    `query` (str): User's query that the AI should respond to.

    Parameters:
        request (Request): A JSON body with `files_path`, `rag_name` and `query`

    Returns:
        A response object containing the response from AI

    Example JSON request body:

    {
        "index_path": "configurations/upload_test_files/iteration_1",
        "rag_name": "base",
        "query": "User's query"

    """
    data = request.get_json()
    working_dir = data.get("index_path")
    rag_name = data.get("rag_name")
    query = data.get("query")
    ragstore_settings = data.get("ragstore_settings")

    use_case_name = working_dir.split("/")[-2]
    iteration = working_dir.split("/")[-1]
    rag_models = ["base", "raptor", "subqa", "meta_llama", "meta_lang", "tableBase"]

    if rag_name not in rag_models:
        return jsonify(
            {
                "error": f"Invalid rag_name, rag_name can be one of this values {rag_models}"
            }
        )

    if not ragstore_settings:
        return jsonify({"error": f"ragstore_settings is required"})

    allowed_rag_settings = {
        "base_rag_setting",
        "subqa_rag_setting",
        "raptor_rag_setting",
        "meta_llama_rag_setting",
        "meta_lang_rag_setting",
        "tableBase_rag_setting",
    }
    if not set(ragstore_settings.keys()).issubset(allowed_rag_settings):
        return jsonify(
            {
                "error": 'ragstore_settings should be one of ["base_rag_setting", "subqa_rag_setting", "raptor_rag_setting", "meta_llama_rag_setting", "meta_lang_rag_setting", "tableBase_rag_setting]'
            }
        )

    try:
        aggrag_ragstore_settings = RagStoreSettings(**ragstore_settings)
        ragstore_settings = RagStoreSettings(
            base_rag_setting=(
                BaseRagSetting(**ragstore_settings.get("base_rag_setting"))
                if ragstore_settings.get("base_rag_setting")
                else None
            ),
            # base_rag_setting=BaseRagSetting(**ragstore_settings.base_rag_setting.model_dump()) if ragstore_settings.base_rag_setting else None,
            raptor_rag_setting=(
                RaptorRagSetting(**ragstore_settings.get("raptor_rag_setting"))
                if ragstore_settings.get("raptor_rag_setting")
                else None
            ),
            subqa_rag_setting=(
                SubQARagSetting(**ragstore_settings.get("subqa_rag_setting"))
                if ragstore_settings.get("subqa_rag_setting")
                else None
            ),
            meta_llama_rag_setting=(
                MetaLlamaRagSetting(**ragstore_settings.get("meta_llama_rag_setting"))
                if ragstore_settings.get("meta_llama_rag_setting")
                else None
            ),
            meta_lang_rag_setting=(
                MetaLangRagSetting(**ragstore_settings.get("meta_lang_rag_setting"))
                if ragstore_settings.get("meta_lang_rag_setting")
                else None
            ),
            tableBase_rag_setting=(
                TableBaseRagSetting(**ragstore_settings.get("tableBase_rag_setting"))
                if ragstore_settings.get("tableBase_rag_setting")
                else None
            ),
        )

    except ValidationError as e:
        # Return a JSON response with the validation errors
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        # Handle other exceptions
        return jsonify({'error': str(e)}), 500
    
    aggrag = AggRAG(
        ragstore_bool=RagStoreBool(**{rag_name: True}),
        usecase_name=use_case_name,
        iteration=iteration,
        DATA_DIR=working_dir,
        ragstore_settings=aggrag_ragstore_settings,
    )

    working_dir = os.getcwd()
    print(f"current working dir: {working_dir} ")
    print(f"data dir: {aggrag.BASE_DIR} {aggrag.DATA_DIR} ")
    index_folder = os.listdir(f"{working_dir}{aggrag.BASE_DIR}/index/")[0]
    directory_path = f"{working_dir}{aggrag.BASE_DIR}/index/{index_folder}"
    print(f"directory path: {directory_path}. Is dir: {os.path.isdir(directory_path)}")
    # _ = aggrag.documents_loader(DIR=f"{working_dir}{aggrag.DATA_DIR}")
    try:

        loop = get_event_loop()
        if os.path.isdir(directory_path):
            print(f"The directory '{directory_path}' exists.")
        else:
            print(
                f"Creating index as the directory '{directory_path}' does not exist. "
            )
            _ = aggrag.documents_loader(DIR=f"{working_dir}{aggrag.DATA_DIR}")

            index = asyncio.run(
                aggrag.create_all_index_async(documents=aggrag.documents, exclude=[])
            )

        index = asyncio.run(aggrag.retrieve_all_index_async())

        asyncio.run(aggrag.load_chat_engines())

        chat_answer = loop.run_until_complete(aggrag.ragstore_chat(query=query))
        print(
            f"response recevied from aggrag: type {type(chat_answer)} and respose: {chat_answer}"
        )
        return jsonify({"response": chat_answer})

    except OpenAIError as e:
        # Handle OpenAI API errors
        logger.error(f"OpenAI API error: {str(e)}")
        return jsonify({'error': 'An error occurred while communicating with the OpenAI API.', 'details': str(e)}), 500
    except FileNotFoundError as e:
        return jsonify({'error': str(e)})

    except Exception as e:
        # Handle other exceptions
        logging.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred.', 'details': str(e)}), 500
        
@app.route('/app/deleteiteration', methods=['DELETE'])
def delete_iteration():
    data = request.get_json()
    usecase_folder_name = data.get("folderName")
    iteration_name = data.get("iterationFolder")

    parent_dir = os.path.join(os.getcwd(), "configurations")

    folder_path = os.path.join(parent_dir, usecase_folder_name, iteration_name)

    if os.path.exists(folder_path):
        try:
            shutil.rmtree(folder_path)
            return jsonify({"message": "Iteration deleted successfully"}), 200
        except Exception as e:
            return jsonify({"message": "Failed to delete"}), 500
    else:
        return jsonify({"message": "Folder does not exist."}), 404


@app.route("/app/deleteusecase", methods=["DELETE"])
def delete_usecase():
    data = request.get_json()
    usecase_name = data.get("usecasename")

    if not usecase_name:
        return jsonify({"message": "failed to delete"}), 400

    parent_dir = os.path.join(os.getcwd(), "configurations")
    folder_path = os.path.join(parent_dir, usecase_name)

    if os.path.exists(folder_path):
        try:
            shutil.rmtree(folder_path)
            return jsonify({"message": "Use case deleted successfully"}), 200
        except Exception as e:
            return jsonify({"message": "Failed to delete"}), 500
    else:
        return jsonify({"message": "Folder does not exist."}), 404


@app.route("/app/healthCheck", methods=["GET"])
def get_health_check():
    return jsonify({"message": "ok"}), 200


@app.route('/app/exportFiles', methods=['GET'])
def exportFiles():
    folder_to_zip = request.args.get('folder_path')
    if not folder_to_zip:
        return jsonify({'error': f"folder_path is required"})

    memory_file = zip_directory(folder_to_zip)
    if not memory_file:
        return jsonify({'error': f"Failed to zip provided directory"})

    return send_file(memory_file, mimetype='application/zip', as_attachment=True, download_name='files.zip')


def run_server(host="", port=8000, cmd_args=None):
    global HOSTNAME, PORT
    HOSTNAME = host
    PORT = port
    app.run(host=host, port=port, debug=True)


def get_event_loop():
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("Event loop is closed")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop

if __name__ == "__main__":
    print("Run app.py instead.")