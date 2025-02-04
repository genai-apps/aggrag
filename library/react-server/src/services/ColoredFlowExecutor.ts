/**
 * ColoredFlowExecutor Class
 *
 * This class is designed to execute nodes in a flow graph where edges are marked as "colored" to indicate special processing paths.
 * It supports parallel execution of nodes where dependencies allow. This is particularly useful for complex workflows where
 * certain tasks can be performed concurrently, improving efficiency and execution time.
 *
 * Key functionalities:
 * - Separates colored and regular edges during initialization.
 * - Identifies entry and output nodes based on the presence of incoming and outgoing edges.
 * - Executes nodes in topological order based on their dependencies, grouped by levels for parallel execution.
 *
 * Usage:
 * The executor requires a `Flow` object containing nodes and edges on initialization. Nodes will be executed based on the
 * colored edges that define their execution dependencies. Regular edges are ignored for execution order but can be used
 * for other purposes such as data validation or additional side effects in a regular flow context.
 *
 * Example:
 * ```
 * const flow = { nodes: [...], edges: [...] };
 * const executor = new ColoredFlowExecutor(flow);
 * executor.execute().then(context => {
 *   console.log("Execution context:", context);
 * });
 * ```
 *
 * Implementation Details:
 * - `executeNode`: Handles the execution of a single node (see lines 75-89).
 * - `execute`: Orchestrates the overall execution, managing parallelism where applicable (see lines 109-132).
 * - `determineExecutionOrder`: Determines the order of execution, organizing nodes into levels for parallel processing (see lines 186-227).
 *
 * Note: This is an experimental feature and may not fully support all types of graph structures, especially those with complex cyclic dependencies.
 *
 * Therefore, ensure you 'Validate' the flow with 'Validate Flow' CTA inside 'Run Flow' CTA.
 */

// Currently coloredFlowExecutor wont handle parallel flows. Plus its an experimental feature
import { Dict } from "../backend/typing";
import { PromptExecutionService } from "./PromptExecutionService";
import { SplitNodeExecutionService } from "./SplitNodeExecutionService";
import { JoinFormat } from "./../JoinNode";
import { executeJsInNode } from "./CodeExecutionService";

export interface Edge {
  id: string;
  source: string;
  target: string;
  sourceHandle: string;
  targetHandle: string;
  data?: {
    colored?: boolean;
  };
}

export interface Node {
  id: string;
  type: string;
  data: any;
}

export interface Flow {
  nodes: Node[];
  edges: Edge[];
}

export class ColoredFlowExecutor {
  private nodes: Map<string, Node>;
  private coloredEdges: Map<string, Edge[]>;
  private regularEdges: Map<string, Edge[]>;
  private entryNodes: Set<string>;
  private outputNodes: Set<string>;

  constructor(flow: Flow) {
    console.log("Initializing ColoredFlowExecutor with provided flow data.");
    this.nodes = new Map(flow.nodes.map((node) => [node.id, node]));
    this.coloredEdges = new Map();
    this.regularEdges = new Map();
    this.entryNodes = new Set();
    this.outputNodes = new Set();

    // Separate colored and regular edges
    flow.edges.forEach((edge) => {
      const edgeMap = edge.data?.colored
        ? this.coloredEdges
        : this.regularEdges;
      if (!edgeMap.has(edge.source)) {
        edgeMap.set(edge.source, []);
      }
      edgeMap.get(edge.source)!.push(edge);
    });

    this.identifySpecialNodes(flow);

    console.log(`Colored edges:`, this.coloredEdges);
    // console.log(`Regular edges:`, this.regularEdges);
    console.log(`Input nodes:`, this.entryNodes);
    console.log(`Output nodes:`, this.outputNodes);
  }

  private identifySpecialNodes(flow: Flow): void {
    console.log("Identifying entry and output nodes.");
    const hasIncomingEdges = new Set(flow.edges.map((e) => e.target));
    const hasOutgoingEdges = new Set(flow.edges.map((e) => e.source));

    this.nodes.forEach((_, id) => {
      if (!hasIncomingEdges.has(id)) {
        this.entryNodes.add(id);
        console.log(`Node ${id} added as entry node.`);
      }
      if (!hasOutgoingEdges.has(id)) {
        this.outputNodes.add(id);
        console.log(`Node ${id} added as output node.`);
      }
    });
  }

  private async executeNode(
    nodeId: string,
    executionContext: Map<string, any>,
  ): Promise<any> {
    console.log(`Executing node ${nodeId}.`);
    const node = this.nodes.get(nodeId);
    if (!node) throw new Error(`Node ${nodeId} not found`);

    // Execute node based on its type
    const result = await this.executeNodeByType(node, executionContext);
    executionContext.set(nodeId, result);

    console.log(`Node ${nodeId} executed with result:`, result);
    return result;
  }

  private async executeNodeByType(
    node: Node,
    context: Map<string, any>,
  ): Promise<any> {
    console.log(`Executing node by type: ${node.type} for id: ${node.id}`);
    console.log(`The updated node data is:`, node.data);
    console.log(`context is for ${node.id}`, context);

    let visibleFields: Dict<string>;

    switch (node.type) {
      case "textfields":
        // Filter out fields that are marked as not visible
        visibleFields = Object.entries(node.data.fields || {})
          .filter(([key, _]) => {
            const visibility = node.data.fields_visibility || {};
            return visibility[key] !== false; // Only include if not explicitly set to false
          })
          .reduce((acc, [key, value]) => {
            acc[key] = String(value);
            return acc;
          }, {} as Dict<string>);

        return {
          type: "textfields",
          output: visibleFields,
          nodeId: node.id,
        };
      // return {
      //   type: "textfields",
      //   output: node.data.fields,
      //   nodeId: node.id,
      // };

      case "uploadfilefields":
        // For file fields, we want to pass the file paths through
        return {
          type: "uploadfilefields",
          output: node.data.fields || {},
          nodeId: node.id,
        };

      case "split":
        return await this.executeSplitNode(node, context);

      case "join":
      case "joinNode":
        return await this.executeJoinNode(node, context);

      case "evaluator":
      case "processor":
        return await this.executeJavaScriptNode(node, context);

      case "prompt":
      case "promptNode":
        return await this.executePromptNode(node, context);
      // Add other node type handlers here
      default:
        throw new Error(`Unsupported node type: ${node.type}`);
    }
  }

  // For parallel flow;
  public async execute(): Promise<Map<string, any>> {
    console.log("Determining execution order...");
    const executionLevels = this.determineExecutionOrder();
    console.log("Execution levels determined:", executionLevels);

    const executionContext = new Map<string, any>();
    const visited = new Set<string>();

    // Execute nodes level by level
    for (const level of executionLevels) {
      // Execute all nodes in current level in parallel
      await Promise.all(
        level.map((nodeId) =>
          this.executeNode(nodeId, executionContext).then(() =>
            visited.add(nodeId),
          ),
        ),
      );
    }

    console.log("Execution of flow completed.");
    return executionContext;
  }

  private async executeJoinNode(
    node: Node,
    context: Map<string, any>,
  ): Promise<any> {
    console.log("executing join node");
  
    const incomingColoredEdges = Array.from(this.coloredEdges.values())
      .flat()
      .filter((edge) => edge.target === node.id);
  
    // Use default format if none specified
    const format = node.data?.format || node.data?.joinFormat || JoinFormat.NumList;
  
    // Create variables dict for input processing
    const variables: Dict<any> = {};
  
    // Process incoming edges and collect input data
    for (const edge of incomingColoredEdges) {
      const sourceResult = context.get(edge.source);
      if (sourceResult?.output) {
        // Convert object values to array of strings
        if (typeof sourceResult.output === 'object' && !Array.isArray(sourceResult.output)) {
          const texts = Object.values(sourceResult.output).map((val: unknown) => String(val));
          variables[edge.targetHandle] = texts;
        } else {
          variables[edge.targetHandle] = Array.isArray(sourceResult.output) 
            ? sourceResult.output.map((val: unknown) => String(val))
            : [String(sourceResult.output)];
        }
      }
    }
  
    try {
      // Get all texts to join
      const allTexts = Object.values(variables).flat();
      
      // Join texts using the joinTexts function logic
      let joinedText = '';
      if (format === JoinFormat.DubNewLine || format === JoinFormat.NewLine) {
        joinedText = allTexts.join(format);
      } else if (format === JoinFormat.DashedList) {
        joinedText = allTexts.map(t => "- " + t).join("\n");
      } else if (format === JoinFormat.NumList) {
        joinedText = allTexts.map((t, i) => `${i + 1}. ${t}`).join("\n");
      } else if (format === JoinFormat.PyArr) {
        joinedText = JSON.stringify(allTexts);
      } else {
        joinedText = allTexts[0] || '';
      }
  
      return {
        type: "join",
        output: {
          text: joinedText,
          metadata: {},
          llm: undefined
        },
        nodeId: node.id,
        metadata: {
          joinFormat: format,
          groupByVar: node.data.groupByVar || "A",
          groupByLLM: node.data.groupByLLM || "within",
          preservedMetadata: {},
          vars: [],
          metavars: [],
          numLLMs: 0
        },
      };
    } catch (error) {
      if (error instanceof Error) {
        throw new Error(`Failed to execute join node: ${error.message}`);
      }
      throw error;
    }
  }

  private async executeSplitNode(
    node: Node,
    context: Map<string, any>,
  ): Promise<any> {
    console.log("executing split node");
    console.log(`Full context details:`, context);

    // Get all colored edges targeting this node
    const incomingColoredEdges = Array.from(this.coloredEdges.values())
      .flat()
      .filter((edge) => edge.target === node.id);

    // Validate split format exists
    if (!node.data?.splitFormat) {
      throw new Error("Split format not specified in node configuration");
    }

    // Get all nodes from the context
    const allNodes = Array.from(this.nodes.values());

    // Create variables dict for input processing
    const variables: Dict<any> = {};

    // Process incoming edges and collect input data
    for (const edge of incomingColoredEdges) {
      const sourceResult = context.get(edge.source);
      if (sourceResult?.output) {
        variables[edge.targetHandle] = sourceResult.output;
      }
    }

    try {
      const splitService = new SplitNodeExecutionService(node.id);
      const result = await splitService.executeSplitNode(
        node.data.splitFormat,
        variables,
        incomingColoredEdges,
        allNodes,
      );

      // Return in format matching UI for consistency
      return {
        type: "split",
        output: result.splitResults,
        nodeId: node.id,
        metadata: {
          splitFormat: node.data.splitFormat,
          preservedMetadata: result.preservedMetadata,
        },
      };
    } catch (error) {
      if (error instanceof Error) {
        if (error.message.includes("format")) {
          throw new Error(`Split format error: ${error.message}`);
        } else if (error.message.includes("input")) {
          throw new Error(`Split input error: ${error.message}`);
        }
        throw new Error(`Failed to execute split node: ${error.message}`);
      }
      throw error;
    }
  }

  private async executePromptNode(
    node: Node,
    context: Map<string, any>,
  ): Promise<any> {
    console.log("executing prompt node");
    console.log(`Full context details:`, context);

    // Get all edges targeting this node
    const incomingColoredEdges = Array.from(this.coloredEdges.values())
      .flat()
      .filter((edge) => edge.target === node.id);

    // Get all nodes from the context
    const allNodes = Array.from(this.nodes.values());

    // Create variables dict for backward compatibility
    const variables: Dict<any> = {};

    // Prepare RAG data structure similar to PromptNode.tsx
    const ragData: any = {
      p_folder: process.env.P_FOLDER || "",
      i_folder: process.env.I_FOLDER || "",
      query: {},
      uid: [],
    };

    for (const edge of incomingColoredEdges) {
      const sourceResult = context.get(edge.source);
      if (sourceResult?.output) {
        if (edge.targetHandle === "rag_knowledge_base") {
          // Handle RAG data similar to PromptNode.tsx lines 1191-1208
          const fileData = sourceResult.output;
          Object.values(fileData).forEach((filePath: unknown) => {
            try {
              // Extract UID from file path following the same pattern as PromptNode
              if (typeof filePath === "string") {
                const uid = filePath.split("/")[3].split("-")[1];
                const pathParts = filePath.split("/");
                console.log(`File Path parts are: ${pathParts}`);
                ragData.i_folder = pathParts[1];
                ragData.p_folder = pathParts[0];
                ragData.uid.push(uid);
              }
            } catch (err) {
              console.error("Error extracting UID from file path:", err);
            }
          });
        }
        // Store all variables for template processing
        variables[edge.targetHandle] = sourceResult.output;
      }
    }

    const promptService = new PromptExecutionService(node.id);

    try {
      // Pass ragData through variables to maintain the structure queryRAG expects
      variables.__ragData = ragData;

      const result = await promptService.executePromptNode(
        node.data.prompt,
        node.data.llms || [],
        node.data.rags || [],
        variables,
        incomingColoredEdges,
        allNodes,
        node.data.apiKeys,
        undefined, // onProgressChange is optional
      );

      return {
        type: "prompt",
        output: result.responses,
        cache: result.cache,
        nodeId: node.id,
      };
    } catch (error) {
      const message = (error as Error).message;
      throw new Error(`Failed to execute prompt node: ${message}`);
    }
  }

  private async executeColoredFlow(
    currentNodeId: string,
    executionContext: Map<string, any>,
    visited: Set<string>,
  ): Promise<void> {
    if (visited.has(currentNodeId)) return;

    console.log(`Executing colored flow node ${currentNodeId}.`);
    await this.executeNode(currentNodeId, executionContext);
    visited.add(currentNodeId);

    const coloredOutgoingEdges = this.coloredEdges.get(currentNodeId) || [];
    for (const edge of coloredOutgoingEdges) {
      console.log(
        `Following colored edge from ${currentNodeId} to ${edge.target}.`,
      );
      await this.executeColoredFlow(edge.target, executionContext, visited);
    }
  }

  // For parallel flow
  public determineExecutionOrder(): string[][] {
    const connectedNodes = this.getConnectedNodes();
    const inDegree = new Map<string, number>();
    const levelGroups: string[][] = [];

    // Initialize inDegree map
    connectedNodes.forEach((nodeId) => {
      inDegree.set(nodeId, 0);
    });

    // Calculate inDegree for each node
    this.coloredEdges.forEach((edges) => {
      edges.forEach((edge) => {
        if (connectedNodes.has(edge.target)) {
          inDegree.set(edge.target, (inDegree.get(edge.target) || 0) + 1);
        }
      });
    });

    // Group nodes by levels (nodes in same level can be executed in parallel)
    while (inDegree.size > 0) {
      const currentLevel: string[] = [];

      // Find all nodes with zero in-degree
      inDegree.forEach((degree, nodeId) => {
        if (degree === 0) {
          currentLevel.push(nodeId);
        }
      });

      if (currentLevel.length === 0 && inDegree.size > 0) {
        throw new Error("Cycle detected in the graph");
      }

      // Remove processed nodes and update in-degrees
      currentLevel.forEach((nodeId) => {
        inDegree.delete(nodeId);
        const edges = this.coloredEdges.get(nodeId) || [];
        edges.forEach((edge) => {
          if (inDegree.has(edge.target)) {
            inDegree.set(edge.target, inDegree.get(edge.target)! - 1);
          }
        });
      });

      if (currentLevel.length > 0) {
        levelGroups.push(currentLevel);
      }
    }

    return levelGroups;
  }

  // ColoredFlowExecutor.ts
  private async executeJavaScriptNode(
    node: Node,
    context: Map<string, any>,
  ): Promise<any> {
    // Get incoming edges and their data
    const incomingColoredEdges = Array.from(this.coloredEdges.values())
      .flat()
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
  
    // Validate required function exists
    const requiredFunction = node.type === 'evaluator' ? 'evaluate' : 'process';
    const functionRegex = new RegExp(`function\\s+${requiredFunction}\\s*\\(`);
    
    if (!functionRegex.test(node.data.code)) {
      throw new Error(`Missing required ${requiredFunction}() function in ${node.type} node`);
    }
  
    try {
      // Use Node.js execution instead of browser-based
      const result = await executeJsInNode(
        node.id,
        node.data.code,
        responseBatch,
        "response",
        node.type as "evaluator" | "processor"
      );
  
      if (result.error) {
        throw new Error(result.error);
      }
  
      return {
        type: node.type,
        output: result.responses,
        nodeId: node.id
      };
    } catch (error) {
      const message = (error as Error).message;
      throw new Error(`Failed to execute ${node.type} node: ${message}`);
    }
  }

  // private async executeJavaScriptNode(
  //   node: Node,
  //   context: Map<string, any>,
  // ): Promise<any> {
  //   // Get incoming edges and their data
  //   const incomingColoredEdges = Array.from(this.coloredEdges.values())
  //     .flat()
  //     .filter((edge) => edge.target === node.id);

  //   // Get response batch from incoming edges
  //   const responseBatch = [];
  //   for (const edge of incomingColoredEdges) {
  //     if (edge.targetHandle === "responseBatch") {
  //       const sourceResult = context.get(edge.source);
  //       if (sourceResult?.output) {
  //         responseBatch.push(...sourceResult.output);
  //       }
  //     }
  //   }

  //   // Validate required function exists
  //   const requiredFunction = node.type === "evaluator" ? "evaluate" : "process";
  //   const functionRegex = new RegExp(`function\\s+${requiredFunction}\\s*\\(`);

  //   if (!functionRegex.test(node.data.code)) {
  //     throw new Error(
  //       `Missing required ${requiredFunction}() function in ${node.type} node`,
  //     );
  //   }

  //   try {
  //     // Execute the JavaScript code
  //     const result = await executejs(
  //       node.id,
  //       node.data.code,
  //       responseBatch,
  //       "response",
  //       node.type as "evaluator" | "processor", // Type assertion to match expected literal type
  //     );

  //     if (result.error) {
  //       throw new Error(result.error);
  //     }

  //     return {
  //       type: node.type,
  //       output: result.responses,
  //       nodeId: node.id,
  //     };
  //   } catch (error) {
  //     const message = (error as Error).message;
  //     throw new Error(`Failed to execute ${node.type} node: ${message}`);
  //   }
  // }

  private async executeRegularFlow(
    currentNodeId: string,
    executionContext: Map<string, any>,
    visited: Set<string>,
  ): Promise<void> {
    if (visited.has(currentNodeId)) return;

    console.log(`Executing regular flow node ${currentNodeId}.`);
    await this.executeNode(currentNodeId, executionContext);
    visited.add(currentNodeId);

    const regularOutgoingEdges = this.regularEdges.get(currentNodeId) || [];
    for (const edge of regularOutgoingEdges) {
      console.log(
        `Following regular edge from ${currentNodeId} to ${edge.target}.`,
      );
      await this.executeRegularFlow(edge.target, executionContext, visited);
    }
  }

  private getConnectedNodes(): Set<string> {
    const connectedNodes = new Set<string>();
    this.coloredEdges.forEach((edges, source) => {
      connectedNodes.add(source);
      edges.forEach((edge) => {
        connectedNodes.add(edge.target);
      });
    });
    return connectedNodes;
  }
}
