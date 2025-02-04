import {
  Dict,
  EvaluatedResponsesResults,
  PythonInterpreter,
  LLMResponse,
  TemplateVarInfo,
  VarsContext,
  QueryProgress
} from "../backend/typing";

import { ResponseInfo, run_over_responses } from "../backend/backend";



// works but does not use run_over_response
export async function executeJsInNode(
  id: string,
  code: string,
  responses: LLMResponse[],
  scope: "response" | "batch",
  processType: "evaluator" | "processor"
): Promise<EvaluatedResponsesResults> {
  const vm = require('vm');
  let context: any;
  
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
      }
    });

    // First compile the user's function
    const funcScript = new vm.Script(`
      ${code}
      (function(response) {
        return ${processType === "evaluator" ? "evaluate" : "process"}(response);
      })
    `);
    const processFunc = funcScript.runInContext(context);

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

    return { responses: processedResponses };
    
  } catch (error) {
    return {
      error: `Execution error: ${error instanceof Error ? error.message : String(error)}`,
      logs: context?.console?.getLogs?.() || []
    };
  }
}

// export async function executeJsInNode(
//   id: string,
//   code: string,
//   responses: LLMResponse[],
//   scope: "response" | "batch",
//   processType: "evaluator" | "processor"
// ): Promise<EvaluatedResponsesResults> {
//   const vm = require('vm');
//   let context: any; // Declare context outside try block
  
//   try {
    
//     // Create secure context with full response structure
//     const context = vm.createContext({
//       console: console,
//       ResponseInfo: function(response: any) {
//         const responseText = response.responses?.[0] || response.text || '';
        
//         // Match UI response structure from backend.ts
//         this.text = String(responseText);
//         this.prompt = String(response.prompt || '');
//         this.var = response.vars || {};
//         this.meta = response.metavars || {};
//         this.llm = response.llm || {};
//         this.eval_res = response.eval_res || {};
        
//         // Add helper methods like in UI
//         this.toString = function() { return this.text; };
//         this.getLLMConfig = function() { return this.llm; };
//         this.getMetadata = function() { return this.meta; };
//       }
//     });

//     // Process responses with proper error handling
//     const processedResponses = responses.map(response => {
//       try {
//         const script = new vm.Script(`
//           ${code}
//           const responseInfo = new ResponseInfo(${JSON.stringify(response)});
//           ${processType === "evaluator" ? "evaluate" : "process"}(responseInfo);
//         `);
//         const result = script.runInContext(context);
        
//         return {
//           ...response,
//           [processType === "evaluator" ? "eval_res" : "text"]: result
//         };
//       } catch (err) {
//         throw new Error(`Processing failed for response: ${err instanceof Error ? err.message : String(err)}`);
      
//       }
//     });

//     return { responses: processedResponses };
    
//   } catch (error) {
//     return {
//       error: `Execution error: ${(error as Error).message}`,
//       logs: context?.console?.getLogs?.() || []
//     };
//   }
// }
