// library/react-server/src/services/PromptExecutionService.ts
import {
  LLMSpec,
  QueryProgress,
  LLMResponse,
  Dict,
  TemplateVarInfo,
} from "../backend/typing";
import { PromptPipeline } from "../backend/query";
import {
  queryLLM,
  queryRAG,
  caching_responses,
  fetchEnvironAPIKeys,
} from "../backend/backend";
import { set_api_keys } from "../backend/utils";
import { Node, Edge } from "./ColoredFlowExecutor";

export class PromptExecutionService {
  constructor(private nodeId: string) {
    console.log(`Executing prompt node for ID: ${nodeId}`); // Utilizing nodeId for logging
  }

  private extractTemplateVariables(template: string): string[] {
    // Match both {var} and {=var} patterns
    const regex = /\{=?([^}]+)\}/g;
    const variables: string[] = [];
    let match;

    while ((match = regex.exec(template)) !== null) {
      variables.push(match[1].trim());
    }

    return variables;
  }

  private processTemplate(template: string, variables: Dict<any>): string {
    let processedTemplate = template;

    // Handle both {var} and {=var} patterns
    Object.entries(variables).forEach(([key, value]) => {
      const patterns = [`{${key}}`, `{=${key}}`];
      patterns.forEach((pattern) => {
        if (processedTemplate.includes(pattern)) {
          let replacementValue = value;
          if (typeof value === "object" && value.output) {
            replacementValue = value.output;
          }
          processedTemplate = processedTemplate.replace(
            pattern,
            String(replacementValue),
          );
        }
      });
    });

    return processedTemplate;
  }

  private pullInputData(
    templateVars: string[],
    nodes: Node[],
    edges: Edge[],
  ): Dict<(string | TemplateVarInfo)[]> {
    const pulled_data: Dict<(string | TemplateVarInfo)[]> = {};

    // Add rag_knowledge_base to template vars if it exists in variables
    const allVars = new Set(templateVars);
    const ragEdge = edges.find(
      (e) =>
        e.targetHandle === "rag_knowledge_base" && e.target === this.nodeId,
    );
    if (ragEdge) {
      allVars.add("rag_knowledge_base");
    }

    Array.from(allVars).forEach((varName) => {
      // Find all edges targeting this variable name
      const sourceEdges = edges.filter(
        (e) => e.target === this.nodeId && e.targetHandle === varName,
      );

      pulled_data[varName] = [];

      sourceEdges.forEach((sourceEdge) => {
        const sourceNode = nodes.find((n) => n.id === sourceEdge.source);
        if (varName === "rag_knowledge_base" && sourceNode?.data?.fields) {
          // Handle RAG data specifically
          const ragPaths = Object.values(sourceNode.data.fields)
            .filter((value) => value !== undefined)
            .map((value) => String(value));
          if (ragPaths.length > 0) {
            pulled_data[varName].push(...ragPaths);
          }
        } else if (sourceNode?.data?.fields) {
          // Regular field handling
          const validFields = Object.entries(sourceNode.data.fields)
            .filter(([key, value]) => {
              const visibility = sourceNode.data.fields_visibility || {};
              if (visibility[key] === false) return false;
              const strValue = String(value);
              return (
                !strValue.includes("{@") &&
                !strValue.includes("{=") &&
                value !== undefined
              );
            })
            .map(([_, value]) => String(value) as string | TemplateVarInfo);

          if (validFields.length > 0) {
            pulled_data[varName].push(...validFields);
          }
        }
      });
    });

    return pulled_data;
  }

  // private pullInputData(
  //   templateVars: string[],
  //   nodes: Node[],
  //   edges: Edge[],
  // ): Dict<(string | TemplateVarInfo)[]> {
  //   const pulled_data: Dict<(string | TemplateVarInfo)[]> = {};

  //   templateVars.forEach((varName) => {
  //     const sourceEdge = edges.find(
  //       (e) => e.target === this.nodeId && e.targetHandle === varName,
  //     );

  //     if (sourceEdge) {
  //       const sourceNode = nodes.find((n) => n.id === sourceEdge.source);
  //       if (sourceNode?.data?.fields) {
  //         // Filter out template variables and respect field visibility
  //         const validFields = Object.entries(sourceNode.data.fields)
  //           .filter(([key, value]) => {
  //             // Check if field is visible based on fields_visibility
  //             const visibility = sourceNode.data.fields_visibility || {};
  //             if (visibility[key] === false) {
  //               return false;
  //             }

  //             // Check if value is a template variable or undefined
  //             const strValue = String(value);
  //             return (
  //               !strValue.includes("{@") &&
  //               !strValue.includes("{=") &&
  //               value !== undefined
  //             );
  //           })
  //           .map(([_, value]) => String(value) as string | TemplateVarInfo);

  //         if (validFields.length > 0) {
  //           pulled_data[varName] = validFields;
  //         }
  //       }
  //     }
  //   });

  //   return pulled_data;
  // }

  async executePromptNode(
    promptTemplate: string,
    llmSpecs: LLMSpec[],
    ragSpecs: LLMSpec[],
    variables: Dict<any>,
    edges: Edge[],
    nodes: Node[],
    apiKeys?: Dict<string>,
    onProgressChange?: (progress: Dict<QueryProgress>) => void,
  ) {
    // If apiKeys not provided, fetch from environment

    if (!apiKeys) {
      apiKeys = await fetchEnvironAPIKeys();
    }
    if (apiKeys) {
      set_api_keys(apiKeys);
    }
    // Log incoming variables for debugging
    console.log("Incoming variables:", variables);

    // Reuse handleRunClick logic for consistency
    const templateVars = this.extractTemplateVariables(promptTemplate);
    console.log("templateVars:");
    console.log(templateVars);
    // Check connections (similar to handleRunClick)
    const is_fully_connected = templateVars.every((varname) => {
      return edges.some(
        (e) => e.target === this.nodeId && e.targetHandle === varname,
      );
    });

    if (!is_fully_connected) {
      throw new Error("Missing inputs to one or more template variables.");
    }

    // Pull input data (similar to handleRunClick's pullInputData)
    // 3. Pull input data and process it
    // Pull input data and process it
    let pulled_data: Dict<(string | TemplateVarInfo)[]> = {};
    try {
      pulled_data = this.pullInputData(templateVars, nodes, edges);

      // If we have no valid data for any template variable, throw an error
      if (Object.keys(pulled_data).length === 0) {
        throw new Error("No valid input data available for template variables");
      }

      console.log("pulled_data");
      console.log(pulled_data);
    } catch (err) {
      throw new Error(`Failed to pull input data: ${(err as Error).message}`);
    }

    // Initialize responses arrays (similar to handleRunClick)
    let LlmRagJsonResponses: LLMResponse[] = [];
    let LlmRagCacheFiles: Dict<string | LLMSpec> = {};

    // Validate LLM/RAG selection
    if (llmSpecs.length === 0 && ragSpecs.length === 0) {
      throw new Error("Please select at least one LLM or RAG to prompt.");
    }

    // Execute queries (similar to handleRunClick's query_llms and query_rags)
    if (llmSpecs?.length > 0) {
      const llmResult = await queryLLM(
        this.nodeId,
        llmSpecs,
        1, // numGenerations
        promptTemplate,
        pulled_data,
        undefined, // chat_hist_by_llm
        apiKeys || {},
        false,
        onProgressChange,
      );

      if (llmResult.responses) {
        LlmRagJsonResponses = llmResult.responses;
      }
      if (llmResult.cache) {
        LlmRagCacheFiles = llmResult.cache;
      }
    }

    // Execute RAG queries if present
    if (ragSpecs?.length > 0) {
      const ragPath = pulled_data.rag_knowledge_base?.[0] as string;
      const pathParts = ragPath.split("/");
      console.log(`Path parts are: ${pathParts}`);
      const ragData = variables.__ragData || {
        p_folder: pathParts[0],
        i_folder: pathParts[1],
        query: pulled_data,
        uid: [],
      };

      ragData.query = pulled_data; // Use pulled_data for the query

      const ragResult = await queryRAG(
        this.nodeId,
        ragSpecs,
        promptTemplate,
        ragData,
        [],
        false,
        onProgressChange,
      );

      if (ragResult.responses) {
        LlmRagJsonResponses = LlmRagJsonResponses.concat(ragResult.responses);
      }
      if (ragResult.cache) {
        LlmRagCacheFiles = { ...LlmRagCacheFiles, ...ragResult.cache };
      }
    }

    // Cache responses (similar to handleRunClick)
    await caching_responses(LlmRagJsonResponses, LlmRagCacheFiles, this.nodeId);

    return {
      responses: LlmRagJsonResponses,
      cache: LlmRagCacheFiles,
    };

    // // Pre-process variables to handle nested structures
    // const processedVars = Object.entries(variables).reduce(
    //   (acc, [key, value]) => {
    //     if (typeof value === "object" && value !== null) {
    //       // Handle nested objects, particularly from text fields
    //       acc[key] = value.output || value;
    //     } else {
    //       acc[key] = value;
    //     }
    //     return acc;
    //   },
    //   {} as Dict<any>,
    // );

    // const processedPrompt = this.processTemplate(promptTemplate, processedVars);
    // console.log("Processed prompt:", processedPrompt);

    // // const processedPrompt = this.processTemplate(promptTemplate, variables);
    // // console.log("Initial processed prompt:", processedPrompt);

    // // Validate template variables
    // const remainingVars = this.extractTemplateVariables(processedPrompt);
    // if (remainingVars.length > 0) {
    //   const missingVars = remainingVars.join(", ");
    //   console.log("Available variables:", Object.keys(processedVars));
    //   throw new Error(
    //     `Template variables not replaced: ${missingVars}. Available variables: ${Object.keys(processedVars).join(", ")}`,
    //   );
    // }

    // const pipeline = new PromptPipeline(processedPrompt);
    // let jsonResponses: LLMResponse[] = [];
    // let cacheFiles: Dict<string | LLMSpec> = {};

    // // console.log("llm specs", llmSpecs);
    // console.log("Executing with variables:", variables);

    // // console.log("LLM Specifications:", llmSpecs);
    // // console.log("RAG Specifications:", ragSpecs);

    // // Execute LLM queries
    // if (llmSpecs?.length > 0) {
    //   const llmResult = await queryLLM(
    //     this.nodeId,
    //     llmSpecs,
    //     1, // numGenerations
    //     processedPrompt,
    //     // promptTemplate,
    //     variables,
    //     undefined, // chat_hist_by_llm
    //     apiKeys || {},
    //     false,
    //     onProgressChange,
    //   );

    //   if (llmResult.responses) {
    //     jsonResponses = llmResult.responses;
    //   }
    //   if (llmResult.cache) {
    //     cacheFiles = llmResult.cache;
    //   }
    // }

    // // Execute RAG queries
    // if (ragSpecs?.length > 0) {
    //   const ragResult = await queryRAG(
    //     this.nodeId,
    //     ragSpecs,
    //     processedPrompt,
    //     // promptTemplate,
    //     variables,
    //     [],
    //     false,
    //     onProgressChange,
    //   );

    //   if (ragResult.responses) {
    //     jsonResponses = jsonResponses.concat(ragResult.responses);
    //   }
    //   if (ragResult.cache) {
    //     cacheFiles = { ...cacheFiles, ...ragResult.cache };
    //   }
    // }

    // // Cache responses
    // await caching_responses(jsonResponses, cacheFiles, this.nodeId);

    // return {
    //   responses: jsonResponses,
    //   cache: cacheFiles,
    // };
  }
}
