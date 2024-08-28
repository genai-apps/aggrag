export const setHintSteps = (
  triggerHint: string,
  hintRuns: any, // replace 'any' with the correct type if known
  setUpdateSteps: (
    selector: string,
    title: string,
    content: string | JSX.Element,
    placement: string,
    disableBeacon: boolean,
    showCloseButton: boolean,
  ) => void,
) => {
  switch (triggerHint) {
    case "created-usecase":
    case "created-iteration":
      setUpdateSteps(
        ".add-node",
        "Hint",
        "Add a Input Data/Knowledge Base node from the list to get started with your work.",
        "left",
        true,
        false,
      );
      break;

    case "textFieldsNode":
      setUpdateSteps(
        ".text-fields-node-for-hint",
        "Hint",
        "Specify your input text here. You can also declare variables in brackets to chain TextFields together.",
        "left",
        true,
        true,
      );

      break;

    case "textfields2":
      if (hintRuns.textField <= 1) {
        setUpdateSteps(
          ".add-node",
          "Hint",
          "Now add a prompt node.",
          "bottom",
          true,
          true,
        );
      }
      break;

    case "textfields3":
      setUpdateSteps(
        ".text-fields-node-for-hint",
        "Hint",
        "You can connect the TextFields node to the prompt node, to get going.",
        "bottom",
        true,
        true,
      );
      break;

    case "promptNode":
      if (hintRuns.prompt <= 1) {
        setUpdateSteps(
          ".prompt-field-fixed-for-hint",
          "Hint",
          <div>
            <div>You can add variables in the node like below:</div>
            <br />
            <span style={{ fontStyle: "italic", color: "#098BCB" }}>
              {"{variable_name}"}
            </span>
          </div>,
          "right",
          true,
          true,
        );
      }
      break;

    case "file-upload":
      if (hintRuns.uploadfilefields <= 1) {
        setUpdateSteps(
          ".file-fields-node",
          "Hint",
          "'Add a RAG' to connect the FileFields Node to the Prompt Node, first. Then, connect the nodes and select 'Create Index'.",
          "bottom",
          true,
          true,
        );
      }
      break;

    case "model-added":
      if (hintRuns.model_added <= 1) {
        setUpdateSteps(
          ".settings-class",
          "Hint",
          "Dont’t forget to add the associated API keys for the LLM Models you have added.",
          "bottom",
          true,
          true,
        );
      }
      break;

    case "prompt-play":
      if (hintRuns.prompthitplay <= 1) {
        setUpdateSteps(
          ".add-node",
          "Hint",
          "Add an Evaluator or Visualizer Node to evaluate/inspect/visualize the responses further.",
          "bottom",
          true,
          true,
        );
      }
      break;

    case "csvNode":
      setUpdateSteps(
        ".items-node-for-hint",
        "Hint",
        "Specify inputs as a comma-separated list of items. Good for specifying lots of short values. A potential alternative to TextFields node.",
        "left",
        true,
        false,
      );
      break;

    case "table":
      setUpdateSteps(
        ".editable-table",
        "Hint",
        "Import or create a spreadsheet of data to use as input to prompt or chat nodes. You can upload xlsx, csv and json.",
        "left",
        true,
        false,
      );
      break;

    case "chatTurn":
      setUpdateSteps(
        ".chat-history",
        "Hint",
        <div>
          <div>Start or continue a conversation with chat models.</div>
          <div>
            Attach Prompt Node output as past context to continue chatting past
            the first turn.
          </div>
        </div>,
        "left",
        true,
        false,
      );
      break;

    case "simpleEval":
      setUpdateSteps(
        ".evaluator-node",
        "Hint",
        "Evaluate responses with a simple check (no coding needed).",
        "left",
        true,
        false,
      );
      break;

    case "evalNode":
      setUpdateSteps(
        ".evaluator-node",
        "Hint",
        "Evaluate responses by writing Python code.",
        "left",
        true,
        false,
      );
      break;

    case "llmeval":
      setUpdateSteps(
        ".llm-eval-node-for-hint",
        "Hint",
        "Evaluate responses with an LLM like GPT-4.",
        "left",
        true,
        false,
      );
      break;

    case "visNode":
      setUpdateSteps(
        ".vis-node",
        "Hint",
        "Plot evaluation results (attach an evaluator or scorer node as input).",
        "left",
        true,
        false,
      );
      break;

    default:
      break;
  }
};
