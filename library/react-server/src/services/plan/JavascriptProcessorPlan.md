# JavaScript Processor/Evaluator Node - API Implementation Plan

## Current Scope for API flow:

To make code work across both flows, the code should:
- Be synchronous
- Avoid DOM/browser APIs
- Return simple data types
- Use only basic console.log
- Follow the examples shown in:

```
export const INFO_EXAMPLE_JS = `
function evaluate(response) {
  // Return the length of the response (num of characters)
  return response.text.length;
}`;
export const INFO_EXAMPLE_VAR_PY = `
def evaluate(response):
  country = response.var['country'];
  # do something with country here, such as lookup whether 
  # the correct capital is in response.text
  return ... # for instance, True or False
`;
export const INFO_EXAMPLE_VAR_JS = `
function evaluate(response) {
  let country = response.var['country'];
  // do something with country here, such as lookup whether 
  // the correct capital is in response.text
  return ... // for instance, true or false
}`;

// Code processor examples for info modal
const INFO_PROC_EXAMPLE_PY = `
def process(response):
  # Return the first 12 characters
  return response.text[:12]
`;
const INFO_PROC_EXAMPLE_JS = `
function process(response) {
  // Return the first 12 characters
  return response.text.slice(0, 12);
}`;
const INFO_PROC_EXAMPLE_VAR_PY = `
def process(response):
  # Find the index of the substring "ANSWER:"
  answer_index = response.text.find("ANSWER:")

  # If "ANSWER:" is in the text, return everything after it
  if answer_index != -1:
    return response.text[answer_index + len("ANSWER:"):]
  else: # return error message
    return "NOT FOUND"
`;
const INFO_PROC_EXAMPLE_VAR_JS = `
function process(response) {
  // Find the index of the substring "ANSWER:"
  const answerIndex = response.text.indexOf("ANSWER:");

  // If "ANSWER:" is in the text, return everything after it
  if (answerIndex !== -1)
    return response.text.substring(answerIndex + "ANSWER:".length);
  else  // return error message
    return "NOT FOUND";
}`;
```

## Current Implementation


1. **Core Execution Logic**

```16:74:library/react-server/src/services/CodeExecutionService.ts
export async function executeJsInNode(
  id: string,
  code: string,
  responses: LLMResponse[],
  scope: "response" | "batch",
  processType: "evaluator" | "processor"
): Promise<EvaluatedResponsesResults> {
  const vm = require('vm');
  let context: any;
      console: console,
  try {
    // Create base context with ResponseInfo class
    context = vm.createContext({
      console: console,
      ResponseInfo: function(response: any) {
        const responseText = response.responses?.[0] || response.text || '';
        this.text = String(responseText);
        this.prompt = String(response.prompt || '');
        this.var = response.vars || {};
        this.meta = response.metavars || {};
        this.llm = response.llm || {};
        this.eval_res = response.eval_res || {};
        this.toString = function() { return this.text; };
        this.getLLMConfig = function() { return this.llm; };
        this.getMetadata = function() { return this.meta; };
        };
    });
      // Export the function
    // First compile the user's function
    const funcScript = new vm.Script(`
      ${code}
      (function(response) {
        return ${processType === "evaluator" ? "evaluate" : "process"}(response);
      })
    `);
    const processFunc = funcScript.runInContext(context);
    const processedResponses = responses.map(response => {
    // Process each response in its own context
    const processedResponses = responses.map(response => {
      try {
        const result = processFunc(new context.ResponseInfo(response));
        return {
          ...response,
          [processType === "evaluator" ? "eval_res" : "text"]: result
        };
      } catch (err) {
        throw new Error(`Processing failed for response: ${err instanceof Error ? err.message : String(err)}`);
      }
    });
    const processedResponses = processFunc(responses);
    return { responses: processedResponses };
    return {
  } catch (error) {
    return {
      error: `Execution error: ${error instanceof Error ? error.message : String(error)}`,
      logs: context?.console?.getLogs?.() || []
    };
  } catch (error) {
    };
```

- Uses Node.js VM for code execution
- Implements ResponseInfo class similar to UI
- Handles single and multiple response processing
- Basic error handling and logging



2. **Node Execution in Flow**

```368:419:library/react-server/src/services/ColoredFlowExecutor.ts
  private async executeJavaScriptNode(
    node: Node,
    context: Map<string, any>,
  ): Promise<any> {
    // Get incoming edges and their data
    const incomingColoredEdges = Array.from(this.coloredEdges.values())
      .flat()
      .filter((edge) => edge.target === node.id);
      .filter((edge) => edge.target === node.id);
    // Get response batch from incoming edges
    const responseBatch = [];
    for (const edge of incomingColoredEdges) {
      if (edge.targetHandle === 'responseBatch') {
        const sourceResult = context.get(edge.source);
        if (sourceResult?.output) {
          responseBatch.push(...sourceResult.output);
        }
        }
      }
    }
    // Validate required function exists
    const requiredFunction = node.type === 'evaluator' ? 'evaluate' : 'process';
    const functionRegex = new RegExp(`function\\s+${requiredFunction}\\s*\\(`);
    const functionRegex = new RegExp(`function\\s+${requiredFunction}\\s*\\(`);
    if (!functionRegex.test(node.data.code)) {
      throw new Error(`Missing required ${requiredFunction}() function in ${node.type} node`);
    }
        `Missing required ${requiredFunction}() function in ${node.type} node`,
    try {
      // Use Node.js execution instead of browser-based
      const result = await executeJsInNode(
        node.id,
        node.data.code,
        responseBatch,
        "response",
        node.type as "evaluator" | "processor"
      );
        "response",
      if (result.error) {
        throw new Error(result.error);
      }
      if (result.error) {
      return {
        type: node.type,
        output: result.responses,
        nodeId: node.id
      };
    } catch (error) {
      const message = (error as Error).message;
      throw new Error(`Failed to execute ${node.type} node: ${message}`);
    } catch (error) {
  }
```

- Handles incoming edges and response batches
- Validates required functions
- Executes code through executeJsInNode



## Limitations


1. **Sandboxing**


- UI: Uses iframe sandbox

```1061:1072:library/react-server/src/backend/backend.ts
      /*
        To run Javascript code in a psuedo-'sandbox' environment, we
        can use an iframe and run eval() inside the iframe, instead of the current environment.
        This is slightly safer than using eval() directly, doesn't clog our namespace, and keeps
        multiple Evaluate node execution environments separate. 
        
        The Evaluate node in the front-end has a hidden iframe with the following id. 
        We need to get this iframe element. 
      */
      iframe = document.getElementById(`${id}-iframe`);
      if (!iframe)
        throw new Error("Could not find iframe sandbox for evaluator node.");
```

- API: Uses Node.js VM (less secure)



2. **Response Processing**


- UI: Uses run_over_responses with proper type checking

```401:484:library/react-server/src/backend/backend.ts
export async function run_over_responses(
  process_func: (resp: ResponseInfo) => any,
  responses: LLMResponse[],
  process_type: "evaluator" | "processor",
): Promise<LLMResponse[]> {
  const evald_resps: Promise<LLMResponse>[] = responses.map(
    async (_resp_obj: LLMResponse) => {
      // Deep clone the response object
      const resp_obj = JSON.parse(JSON.stringify(_resp_obj));

      // Clean up any escaped braces
      resp_obj.responses = resp_obj.responses.map(cleanEscapedBraces);

      // Whether the processor function is async or not
      const async_processor =
        process_func?.constructor?.name === "AsyncFunction";

      // Map the processor func over every individual response text in each response object
      const res = resp_obj.responses;
      const llm_name = extract_llm_nickname(resp_obj.llm);
      let processed = res.map((r: string) => {
        const r_info = new ResponseInfo(
          r,
          resp_obj.prompt,
          resp_obj.vars,
          resp_obj.metavars || {},
          llm_name,
        );

        // Dynamically detect if process_func is async, and await its response;
        // otherwise, simply execute the function.
        return process_func(r_info);
      });

      // If the processor function is async we still haven't gotten responses; we need to wait for Promises to return:
      // NOTE: For some reason, async_processor check may not work in production builds. To circumvent this,
      //       we also check if 'processed' has a Promise (is it assume all processed items will then be promises).
      if (
        async_processor ||
        (processed.length > 0 && processed[0] instanceof Promise)
      ) {
        processed = await Promise.allSettled(processed);
        for (let i = 0; i < processed.length; i++) {
          const elem = processed[i];
          if (elem.status === "rejected")
            // Bubble up errors
            throw new Error(elem.reason);
          processed[i] = elem.value;
        }
      }
      // If type is just a processor
      if (process_type === "processor") {
        // Replace response texts in resp_obj with the transformed ones:
        resp_obj.responses = processed;
      } else {
        // If type is an evaluator
        // Check the type of evaluation results
        // NOTE: We assume this is consistent across all evaluations, but it may not be.
        const eval_res_type = check_typeof_vals(processed);

        if (eval_res_type === MetricType.Numeric) {
          // Store items with summary of mean, median, etc
          resp_obj.eval_res = {
            items: processed,
            dtype: getEnumName(MetricType, eval_res_type),
          };
        } else if (
          [MetricType.Unknown, MetricType.Empty].includes(eval_res_type)
        ) {
          throw new Error(
            "Unsupported types found in evaluation results. Only supported types for metrics are: int, float, bool, str.",
          );
        } else {
          // Categorical, KeyValue, etc, we just store the items:
          resp_obj.eval_res = {
            items: processed,
            dtype: getEnumName(MetricType, eval_res_type),
          };
        }
      }

      return resp_obj;
    },
```
- API: Custom implementation without full type validation



3. **Console Logging**


- UI: Has sophisticated console hijacking

```1100:1119:library/react-server/src/backend/backend.ts
    // Intercept any calls to console.log, .warn, or .error, so we can store the calls
    // and print them in the 'output' footer of the Evaluator Node:
    // @ts-expect-error undefined
    HIJACK_CONSOLE_LOGGING(id, iframe.contentWindow);

    // Run the user-defined 'evaluate' function over the responses:
    // NOTE: 'evaluate' here was defined dynamically from 'eval' above. We've already checked that it exists.

    processed_resps = await run_over_responses(
      iframe ? process_func : code,
      responses,
      process_type,
    );

    // Revert the console.log, .warn, .error back to browser default:

    all_logs = all_logs.concat(
      // @ts-expect-error undefined
      REVERT_CONSOLE_LOGGING(id, iframe.contentWindow),
    );
```

- API: Basic console access





## Limitations


### 1. Security
- No proper sandboxing mechanism
- Direct VM execution could be risky
- Limited code validation

### 2. Functionality
- No async function support
- Missing proper type checking for evaluation results
- No support for batch processing mode
- Limited error context in responses
- 

### 3. Integration
- Cannot use UI-specific features (iframe, DOM)
- No progress reporting mechanism
- Limited console logging capabilities

### 4. Response Handling
- Does not handle all response types (e.g., MetricType validation)
- Missing proper response structure validation
- No support for complex evaluation results



## Recommended Improvements



1. **Security**
- Implement proper sandboxing
- Add code validation
- Add input sanitization



2. **Functionality**
 
- Add async function support
- Implement proper type checking
- Add batch processing support


3. **Integration**
- Add proper response validation


4. **Response Handling**

- Implement MetricType validation
- Add proper response structure validation
- Support complex evaluation results