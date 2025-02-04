#### 1. Create JoinNodeExecutionService Class

```typescript:library/react-server/src/services/JoinNodeExecutionService.ts
// ... existing imports ...
import { JoinFormat } from "../JoinNode"; // Import existing enum

export class JoinNodeExecutionService {
  // Mirror core joining logic from UI implementation
  private joinTexts(texts: string[], format: JoinFormat): string {
    // ... implement joining logic ...
  }

  async execute(node: Node, context: Map<string, any>): Promise<ExecutionResult> {
    // Get colored incoming edges
    // Process inputs maintaining metadata
    // Apply join format from node.data
    // Return standardized output
  }
}
```

#### 2. Add Join Node Support to ColoredFlowExecutor

```typescript:library/react-server/src/services/ColoredFlowExecutor.ts
// ... existing code ...
private async executeNodeByType(node: Node, context: Map<string, any>) {
  switch (node.type) {
    // ... existing cases ...
    case "join":
      return await this.executeJoinNode(node, context);
    // ... existing cases ...
  }
}

private async executeJoinNode(node: Node, context: Map<string, any>) {
  const service = new JoinNodeExecutionService();
  return await service.execute(node, context);
}
```

#### 3. Implement Standard Response Format

```typescript:library/react-server/src/services/typing.ts
interface JoinNodeOutput {
  type: "join";
  output: string | TemplateVarInfo;
  nodeId: string;
  metadata: {
    joinFormat: JoinFormat;
    originalMetadata: Dict<any>;
  }
}
```

#### 4. Implement toString/toJSON Methods

```typescript:library/react-server/src/services/JoinNodeExecutionService.ts
class JoinNodeResponse implements TemplateVarInfo {
  constructor(
    public text: string,
    public metadata: Dict<any>,
    public llm?: LLMSpec
  ) {}

  toString(): string {
    return this.text;
  }

  toJSON(): any {
    return {
      text: this.text,
      metadata: this.metadata,
      llm: this.llm
    };
  }
}
```

#### 5. Update Execution Context Handling

```typescript:library/react-server/src/services/ColoredFlowExecutor.ts
private async executeNode(nodeId: string, context: Map<string, any>): Promise<void> {
  // ... existing code ...

  // Add special handling for join nodes to properly merge metadata
  if (node.type === 'join') {
    const incomingEdges = this.getColoredIncomingEdges(nodeId);
    const inputs = this.gatherInputs(incomingEdges, context);
    // Preserve metadata from all inputs
  }
}
```

#### 6. Testing Tasks

1. Test basic joining functionality

- All format types
- Single and multiple inputs
- Empty/null handling

2. Test metadata preservation

- LLM information
- Template variables
- Colored edge handling

3. Test integration

- Connection with prompt nodes
- Connection with split nodes
- Multi-node workflows

4. Test error cases

- Missing inputs
- Invalid format specifications
- Malformed metadata

#### 7. Documentation Updates

````markdown:library/react-server/src/services/README.md
## Join Node Execution

The join node supports the following formats:
- Double newline (\n\n)
- Single newline (\n)
- Dashed list
- Numbered list
- Python array format


