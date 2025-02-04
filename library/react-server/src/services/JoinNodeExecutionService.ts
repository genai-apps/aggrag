import { Node, Edge } from "./ColoredFlowExecutor";
import { JoinFormat } from "./../JoinNode";
import {
  Dict,
  TemplateVarInfo,
  LLMSpec,
  LLMResponsesByVarDict,
} from "../backend/typing";
import {
  tagMetadataWithLLM,
  extractLLMLookup,
  removeLLMTagFromMetadata,
} from "../backend/utils";
import { escapeBraces } from "../backend/template";

interface ExecutionResult {
  type: "join";
  output: TemplateVarInfo;
  nodeId: string;
  metadata: {
    joinFormat: JoinFormat;
    originalMetadata: Dict<any>;
  };
}

export class JoinNodeExecutionService {
  constructor(private nodeId: string) {
    // Add any initialization logic here to make the constructor useful
    // For example:
    console.log(`Initializing JoinNodeExecutionService for node: ${nodeId}`);
  }

  private joinTexts(texts: string[], format: JoinFormat): string {
    const escaped_texts = texts.map((t) => escapeBraces(t));

    switch (format) {
      case JoinFormat.DubNewLine:
      case JoinFormat.NewLine:
        return escaped_texts.join(format);
      case JoinFormat.DashedList:
        return escaped_texts.map((t) => `- ${t}`).join("\n");
      case JoinFormat.NumList:
        return escaped_texts.map((t, i) => `${i + 1}. ${t}`).join("\n");
      case JoinFormat.PyArr:
        return JSON.stringify(escaped_texts);
      default:
        console.error(`Unknown join format: ${format}`);
        return escaped_texts[0] || "";
    }
  }

  async execute(
    node: Node,
    context: Map<string, any>,
    incomingEdges: Edge[],
  ): Promise<ExecutionResult> {
    // Validate format
    const format = node.data?.format || JoinFormat.DubNewLine;
    if (!Object.values(JoinFormat).includes(format)) {
      throw new Error(`Invalid join format: ${format}`);
    }

    // Gather inputs while preserving metadata
    const inputs: Dict<(string | TemplateVarInfo)[]> = {};
    const llm_lookup: Dict<LLMSpec> = {};

    for (const edge of incomingEdges) {
      const sourceResult = context.get(edge.source);
      if (!sourceResult?.output) continue;

      // Handle both direct string outputs and TemplateVarInfo objects
      const output = sourceResult.output;
      const outputArray = Array.isArray(output) ? output : [output];

      inputs[edge.targetHandle] = outputArray.map((item) => {
        if (typeof item === "string") return item;
        return {
          text: item.text || "",
          metadata: item.metadata || {},
          llm: item.llm,
        };
      });

      // Build LLM lookup table
      if (sourceResult.metadata?.llm) {
        llm_lookup[sourceResult.nodeId] = sourceResult.metadata.llm;
      }
    }

    // Tag metadata with LLM info
    const taggedInputs = tagMetadataWithLLM(inputs) as Dict<
      (string | TemplateVarInfo)[]
    >;

    // Extract texts while preserving metadata
    const textsToJoin: string[] = [];
    const preservedMetadata: Dict<any> = {};

    Object.values(taggedInputs)
      .flat()
      .forEach((item, index) => {
        if (typeof item === "string") {
          textsToJoin.push(item);
        } else {
          textsToJoin.push(item.text || "");
          if (item.metavars || item.llm) {
            preservedMetadata[`input_${index}`] = {
              metadata: item.metavars,
              llm: item.llm,
            };
          }
        }
      });

    // Join texts using specified format
    const joinedText = this.joinTexts(textsToJoin, format);

    // Create response that implements TemplateVarInfo
    const response = {
      text: joinedText,
      metadata: preservedMetadata,
      llm: Object.values(llm_lookup)[0], // Preserve first LLM if available
      toString: function () {
        return this.text;
      },
      toJSON: function () {
        return {
          text: this.text,
          metadata: this.metadata,
          llm: this.llm,
        };
      },
    };

    return {
      type: "join",
      output: response,
      nodeId: this.nodeId,
      metadata: {
        joinFormat: format,
        originalMetadata: preservedMetadata,
      },
    };
  }
}
