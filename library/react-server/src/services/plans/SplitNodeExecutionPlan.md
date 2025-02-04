### 1. Create SplitNodeExecutionService Class ✅

Similar to PromptExecutionService, create a new service class to handle split node execution:

a. Basic structure referencing:

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

b. The service should mirror the core splitting logic from:

const handleOnConnect = useCallback(() => {
const formatting = splitOnFormat;

    let input_data = pullInputData(["__input"], id);
    if (!input_data?.__input) {
      // soft fail if no inputs detected
      return;
    }

    // Create lookup table for LLMs in input, indexed by llm key
    const llm_lookup = extractLLMLookup(input_data);

    // Tag all response objects in the input data with a metavar for their LLM (using the llm key as a uid)
    input_data = tagMetadataWithLLM(input_data) as Dict<
      string[] | TemplateVarInfo[]
    >;

    // Generate (flatten) the inputs, which could be recursively chained templates
    // and a mix of LLM resp objects, templates, and strings.
    // (We tagged each object with its LLM key so that we can use built-in features to keep track of the LLM associated with each response object)
    generatePrompts("{__input}", input_data)
      .then((promptTemplates) => {
        // Convert the templates into response objects
        const resp_objs = promptTemplates.map((p) => ({
          text: p.toString(),
          fill_history: dict_excluding_key(p.fill_history, "__input"),
          llm:
            "__LLM_key" in p.metavars
              ? llm_lookup[p.metavars.__LLM_key]
              : undefined,
          metavars: removeLLMTagFromMetadata(p.metavars),
          uid: uuid(),
        }));

        // The naive splitter is just to look at every
        // response object's text value, and split that into N objects
        // that have the exact same properties except for their text values.
        const split_objs: (TemplateVarInfo | string)[] = resp_objs
          .map((resp_obj: TemplateVarInfo | string) => {
            if (typeof resp_obj === "string")
              return splitText(resp_obj, formatting, true);
            const texts = splitText(resp_obj?.text ?? "", formatting, true);
            if (texts !== undefined && texts.length >= 1)
              return texts.map(
                (t: string) =>
                  deepcopy_and_modify(resp_obj, { text: t }) as TemplateVarInfo,
              );
            else if (texts?.length === 0) return [];
            else return deepcopy(resp_obj) as TemplateVarInfo;
          })
          .flat(); // flatten the split response objects

        setSplitTexts(split_objs);
        setDataPropsForNode(id, { fields: split_objs });
      })

### 2. Add Split Node Support to ColoredFlowExecutor ✅

a. Update the switch case in executeNodeByType method:
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

      case "prompt":
      case "promptNode":
        return await this.executePromptNode(node, context);
      // Add other node type handlers here
      default:
        throw new Error(`Unsupported node type: ${node.type}`);
    }

Add a new case for "split" node type.

### 3. Implement Split Node Execution Logic ✅

a. Create executeSplitNode method in ColoredFlowExecutor that:

- Gets incoming colored edges
- Retrieves input data from context
- Maintains metadata and LLM information
- Uses the saved split format from node data
- Returns split results in standard format

b. The execution should preserve:

- LLM metadata (reference how PromptNode handles it)
- Edge coloring information
- Template variable information

### 4. Handle Parallel Processing ✅

a. No changes needed to execution order logic since it's already handled by:

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

### 5. Update Response Format ✅

a. Standardize split node output format to match other nodes:

{
type: "split",
output: splitResults,
nodeId: node.id,
metadata: {
splitFormat: node.data.splitFormat,
originalMetadata: preservedMetadata
}
}

### 6. Testing Tasks

Test split node execution with:

- Simple text input ✅
- LLM response input with metadata ✅
- Multiple parallel inputs ✅
- Different split formats ✅
- Verify metadata preservation ❌
- Verify colored edge handling ✅
- Test parallel execution ✅
- Validate output format consistency ❌
- Test using API with user input such as `@user_dm` ✅

### 7. Documentation Updates

Currently the following nodes are supported:

- Text Fields Node
- Prompt Node
- File Fields Node

Support for other nodes will be added based on the realised usefulness of the API. Should you need other nodes to be supported, kindly raise an issue or reach out!

2. Document split node API behavior and requirements

### Misc. and Clarifications

1. Core Functionality:

- Handle both string and TemplateVarInfo inputs (like UI)
- Maintain exact parity with UI's metadata handling
- Use splitFormat from node data only (no defaults)
- Use same cache mechanism as UI

2. Error/Edge Cases:

- Missing split format: Show error
- Missing inputs: Soft fail with console error and include in API response
- Missing LLM metadata: Preserve whatever exists, don't add defaults

3. Implementation Strategy:

- Focus on basic functionality first
- Add comments for future code unification
- Use existing cache and metadata handling patterns

### 8. Improvements/Current Bugs: TODO ❌

1. Split Node Output Format Issue: TODO ❌
   When split node output is connected to a prompt node, the prompt template shows "[object Object]" instead of the actual text content. This happens despite implementing toString() and toJSON() methods on the split objects.
