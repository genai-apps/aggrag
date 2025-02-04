import express, { Router } from "express";
import cors from "cors";
import { Node } from "reactflow";
import { ColoredFlowExecutor } from "../services/ColoredFlowExecutor";

const app = express();

// Enable CORS and JSON parsing
app.use(cors());
// app.use(express.json());
app.use(express.json({ limit: "5mb" }));
// Create router for flow endpoints
const router = Router();

router.post("/app/run", async (req, res) => {
  console.log("[/app/run] Received flow execution request");
  try {
    const { flow } = req.body;
    // console.log(`[/app/run] Flow execution request received with flow:`, flow);
    // Validate flow structure
    if (!flow || !flow.nodes || !flow.edges) {
      return res.status(400).json({ error: "Invalid flow structure" });
    }
    console.log(
      `[/app/run] Executing flow with ${flow.nodes.length} nodes and ${flow.edges.length} edges`,
    );
    // Create executor and run flow
    // const executor = new FlowExecutor(flow);
    const executor = new ColoredFlowExecutor(flow);
    const results = await executor.execute();

    // Convert results map to object for response
    const response = Object.fromEntries(results);
    console.log("[/app/run] Flow execution completed successfully");
    res.json({
      success: true,
      results: response,
    });
  } catch (error) {
    // Detailed error logging
    console.error("[/app/run] Flow execution failed:", {
      error: error instanceof Error ? error.message : "Unknown error",
      stack: error instanceof Error ? error.stack : undefined,
      timestamp: new Date().toISOString(),
    });
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : "Unknown error",
    });
  }
});

router.post("/app/validate", async (req, res) => {
  console.log("[/app/validate] Received flow validation request");
  try {
    const { flow } = req.body;

    if (!flow || !flow.nodes || !flow.edges) {
      return res.status(400).json({ error: "Invalid flow structure" });
    }

    // Create executor to analyze the flow
    const executor = new ColoredFlowExecutor(flow);

    // Get execution order and required variables
    const executionLevels = executor.determineExecutionOrder();
    const requiredVariables = new Set<string>();

    // Collect information about each node
    const nodeDetails = executionLevels.map((level) => {
      return level.map((nodeId) => {
        // const node = flow.nodes.find(n => n.id === nodeId);
        const node = flow.nodes.find((n: Node) => n.id === nodeId);

        // For text field nodes, collect required variables
        if (node?.type === "textfields" && node.data?.fields) {
          Object.values(node.data.fields).forEach((field) => {
            const matches = String(field).match(/\{@([^}]+)\}/g);
            if (matches) {
              matches.forEach((match) => {
                requiredVariables.add(match.slice(2, -1));
              });
            }
          });
        }

        return {
          id: nodeId,
          type: node?.type,
          name: node?.data?.title || `${node?.type}-${nodeId}`,
        };
      });
    });

    // Generate example curl command
    // Generate example curl command as bash string
    const vars = Object.fromEntries(
      Array.from(requiredVariables).map((v) => [`@${v}`, "<value>"]),
    );

    const curlExample = `curl -X POST http://localhost:8000/app/run \\
      -H "Content-Type: application/json" \\
      -d '{
        "flow_path": "<path_to_flow_file>",
        "vars": ${JSON.stringify(vars, null, 4)}
      }'`;

    res.json({
      success: true,
      validation: {
        executionOrder: nodeDetails,
        requiredVariables: Array.from(requiredVariables),
        curlExample,
      },
    });
  } catch (error) {
    console.error("[/app/validate] Flow validation failed:", {
      error: error instanceof Error ? error.message : "Unknown error",
      stack: error instanceof Error ? error.stack : undefined,
      timestamp: new Date().toISOString(),
    });

    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : "Unknown error",
    });
  }
});

// Mount the router
app.use("/", router);

export default app;
