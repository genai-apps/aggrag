/**
This implementation:
Mirrors the core splitting logic from SplitNode.tsx (referencing lines 251-305)
Follows the service pattern from PromptExecutionService.ts (referencing lines 19-24)
Maintains metadata handling like ColoredFlowExecutor (referencing lines 206-286)
Key features:
Handles both string and TemplateVarInfo inputs
Preserves LLM metadata and template information
Uses the same cache mechanism as UI
Includes error handling and logging
Supports all split formats from the UI version```
*/

import { Dict, TemplateVarInfo } from "../backend/typing";
import { Node, Edge } from "./ColoredFlowExecutor";
import {
  extractLLMLookup,
  tagMetadataWithLLM,
  deepcopy_and_modify,
  dict_excluding_key,
  removeLLMTagFromMetadata,
} from "../backend/utils";
import { generatePrompts } from "../backend/backend";
import { v4 as uuid } from "uuid";
import StorageCache from "../backend/cache";

export class SplitNodeExecutionService {
  constructor(private nodeId: string) {
    console.log(`Initializing split node service for ID: ${nodeId}`);
  }

  //   async executeSplitNode(
  //     format: string | undefined,
  //     input: Dict<any>,
  //     edges: Edge[],
  //     nodes: Node[],
  //   ): Promise<{
  //     splitResults: (TemplateVarInfo | string)[];
  //     preservedMetadata: any;
  //   }> {
  //     // Validate split format
  //     if (!format) {
  //       throw new Error("Split format must be specified in node data");
  //     }

  //     // Extract input text based on input type
  //     let inputText = "";
  //     let preservedMetadata = {};

  //     if (input.__input) {
  //       if (Array.isArray(input.__input)) {
  //         // Case: Input from prompt node
  //         const promptOutput = input.__input[0];
  //         if (promptOutput?.responses?.[0]) {
  //           inputText = promptOutput.responses[0];
  //           preservedMetadata = {
  //             llm: promptOutput.llm,
  //             vars: promptOutput.vars,
  //             metavars: promptOutput.metavars,
  //             uid: promptOutput.uid
  //           };
  //         }
  //       } else if (typeof input.__input === 'object') {
  //         // Case: Input from text fields node
  //         inputText = input.__input.f1 || '';
  //       } else {
  //         // Case: Direct string input
  //         inputText = String(input.__input);
  //       }
  //     }

  //     if (!inputText) {
  //       throw new Error("No valid input text detected for split node");
  //     }

  //     try {
  //       // Split the text while preserving metadata
  //       const texts = this.splitText(inputText, format);

  //       // Create response objects that will be properly stringified when used as input
  //         const split_objs = texts.map(text => ({
  //             text: text,
  //             fill_history: {},
  //             metavars: preservedMetadata,
  //             uid: uuid(),
  //             toString() {
  //             return this.text;
  //             },
  //             toJSON() {
  //             return this.text;
  //             }
  //         }));

  //       // Cache the results
  //       await StorageCache.store(
  //         `${this.nodeId}.json`,
  //         split_objs
  //       );

  //       return {
  //         splitResults: split_objs,
  //         preservedMetadata: {
  //           original_metadata: preservedMetadata
  //         },
  //       };
  //     } catch (error) {
  //       const message = (error as Error).message;
  //       throw new Error(`Failed to execute split node: ${message}`);
  //     }
  //   }

  async executeSplitNode(
    format: string | undefined,
    input: Dict<any>,
    edges: Edge[],
    nodes: Node[],
  ): Promise<{
    splitResults: (TemplateVarInfo | string)[];
    preservedMetadata: any;
  }> {
    // Validate split format
    if (!format) {
      throw new Error("Split format must be specified in node data");
    }

    // Extract input text based on input type
    let inputText = "";
    let preservedMetadata = {};

    if (input.__input) {
      if (Array.isArray(input.__input)) {
        // Case: Input from prompt node
        const promptOutput = input.__input[0];
        if (promptOutput?.responses?.[0]) {
          inputText = promptOutput.responses[0];
          preservedMetadata = {
            llm: promptOutput.llm,
            vars: promptOutput.vars,
            metavars: promptOutput.metavars,
            uid: promptOutput.uid,
          };
        }
      } else if (typeof input.__input === "object") {
        // Case: Input from text fields node
        inputText = input.__input.f1 || "";
      } else {
        // Case: Direct string input
        inputText = String(input.__input);
      }
    }

    if (!inputText) {
      throw new Error("No valid input text detected for split node");
    }

    try {
      // Create a response object preserving metadata
      const resp_obj = {
        text: inputText,
        fill_history: {},
        metavars: preservedMetadata,
        uid: uuid(),
      };

      // Split the text while preserving metadata
      const texts = this.splitText(resp_obj.text, format);
      const split_objs = texts.map(
        (t) => deepcopy_and_modify(resp_obj, { text: t }) as TemplateVarInfo,
      );

      // Cache the results
      await StorageCache.store(`${this.nodeId}.json`, split_objs);

      return {
        splitResults: split_objs,
        preservedMetadata: {
          original_metadata: resp_obj.metavars,
        },
      };
    } catch (error) {
      const message = (error as Error).message;
      throw new Error(`Failed to execute split node: ${message}`);
    }
  }

  private splitText(text: string, format: string): string[] {
    if (!text) return [];

    switch (format) {
      case "list":
        return text
          .split(/[-*]\s/)
          .map((item) => item.trim())
          .filter(Boolean);
      case "\n":
        return text
          .split("\n")
          .map((item) => item.trim())
          .filter(Boolean);
      case "\n\n":
        return text
          .split("\n\n")
          .map((item) => item.trim())
          .filter(Boolean);
      case ",":
        return text
          .split(",")
          .map((item) => item.trim())
          .filter(Boolean);
      case "code":
        return text.match(/```[\s\S]*?```/g) || [];
      case "paragraph":
        return text
          .split(/\n\s*\n/)
          .map((item) => item.trim())
          .filter(Boolean);
      default:
        throw new Error(`Unsupported split format: ${format}`);
    }
  }
}
