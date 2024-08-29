import React from "react";
export type HintRunsType = {
  promptNode: number;
  textFieldsNode: number;
  uploadFileFieldsNode: number;
  usecase: number;
  iteration: number;
  prompthitplay: number;
  model_added: number;
  csvNode: number;
  table: number;
  chatTurn: number;
  simpleEval: number;
  evalNode: number;
  llmeval: number;
  visNode: number;
  file_upload: number;
  textfields2: number;
  textfields3: number;
};
// this method is to track, how many times a hint is displayed.
export const incrementHintRun = (
  id: string,
  setHintRuns: React.Dispatch<React.SetStateAction<HintRunsType>>,
) => {
  setHintRuns((prevState: HintRunsType) => {
    const key = id as keyof HintRunsType;
    if (key in prevState) {
      const newState = {
        ...prevState,
        [key]: prevState[key] + 1,
      };
      localStorage.setItem("hintRuns", JSON.stringify(newState));
      return newState;
    }
    return prevState;
  });
};
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
  setHintRuns: React.Dispatch<React.SetStateAction<HintRunsType>>,
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
      if (hintRuns && hintRuns.textFieldsNode <= 1) {
        setUpdateSteps(
          ".text-fields-node-for-hint",
          "Hint",
          "Specify your input text here. You can also declare variables in brackets to chain TextFields together.",
          "left",
          true,
          true,
        );
      }
      break;

    case "textfields2":
      // when a textFieldsNode is added, and if user closes then we need to show this hint
      if (hintRuns && hintRuns.textfields2 <= 1) {
        setUpdateSteps(
          ".add-node",
          "Hint",
          "Now add a prompt node.",
          "bottom",
          true,
          true,
        );
        incrementHintRun("textfields2", setHintRuns);
      }
      break;

    case "textfields3":
      // this hint is shown after promptNode
      if (hintRuns && hintRuns.textfields3 < 1) {
        setUpdateSteps(
          ".text-fields-node-for-hint",
          "Hint",
          "You can connect the TextFields node to the prompt node, to get going.",
          "bottom",
          true,
          true,
        );
      }
      incrementHintRun("textfields3", setHintRuns);
      break;

    case "promptNode":
      // for prompt node
      if (hintRuns && hintRuns.promptNode <= 1) {
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
      // when the user uploads file in uploadFileFieldsNode, then we need to show this hint
      if (hintRuns && hintRuns.file_upload <= 0) {
        setUpdateSteps(
          ".file-fields-node",
          "Hint",
          "'Add a RAG' to connect the FileFields Node to the Prompt Node, first. Then, connect the nodes and select 'Create Index'.",
          "bottom",
          true,
          true,
        );
      }
      incrementHintRun("file_upload", setHintRuns);
      break;

    case "model-added":
      // when the user adds a model(Models to query) in Prompt Node, then the following hint will be shown
      if (hintRuns && hintRuns.model_added <= 0) {
        setUpdateSteps(
          ".settings-class",
          "Hint",
          "Dont’t forget to add the associated API keys for the LLM Models you have added.",
          "bottom",
          true,
          true,
        );
      }
      incrementHintRun("model_added", setHintRuns);
      break;

    case "prompt-play":
      // when we click play button in prompt node.
      if (hintRuns && hintRuns.prompthitplay <= 0) {
        setUpdateSteps(
          ".add-node",
          "Hint",
          "Add an Evaluator or Visualizer Node to evaluate/inspect/visualize the responses further.",
          "bottom",
          true,
          true,
        );
      }
      incrementHintRun("prompthitplay", setHintRuns);
      break;

    case "csvNode":
      // CSV node is Items node -- (Input Data)
      if (hintRuns && hintRuns.csvNode <= 1) {
        setUpdateSteps(
          ".items-node-for-hint",
          "Hint",
          "Specify inputs as a comma-separated list of items. Good for specifying lots of short values. A potential alternative to TextFields node.",
          "left",
          true,
          false,
        );
      }
      break;

    case "table":
      // for Tabular Data Node - (Input Data Node)
      if (hintRuns && hintRuns.table <= 1) {
        setUpdateSteps(
          ".editable-table",
          "Hint",
          "Import or create a spreadsheet of data to use as input to prompt or chat nodes. You can upload xlsx, csv and json.",
          "left",
          true,
          false,
        );
      }
      break;

    case "chatTurn":
      // for chat turn node
      if (hintRuns && hintRuns.chatTurn <= 1) {
        setUpdateSteps(
          ".chat-history",
          "Hint",
          <div>
            <div>Start or continue a conversation with chat models.</div>
            <div>
              Attach Prompt Node output as past context to continue chatting
              past the first turn.
            </div>
          </div>,
          "left",
          true,
          false,
        );
      }
      break;

    case "simpleEval":
      // for simple evalutor node
      if (hintRuns && hintRuns.simpleEval <= 1) {
        setUpdateSteps(
          ".simple-eval-for-hint",
          "Hint",
          "Evaluate responses with a simple check (no coding needed).",
          "left",
          true,
          false,
        );
      }
      break;

    case "evalNode":
      // for python node
      // for javascript we need to do additional code here
      setUpdateSteps(
        ".python-cls-for-hint",
        "Hint",
        "Evaluate responses by writing Python code.",
        "bottom",
        true,
        false,
      );
      break;

    case "llmeval":
      // for LLM Scorer
      if (hintRuns && hintRuns.llmeval <= 1) {
        setUpdateSteps(
          ".llm-eval-node-for-hint",
          "Hint",
          "Evaluate responses with an LLM like GPT-4.",
          "left",
          true,
          false,
        );
      }
      break;

    case "visNode":
      // for visualizer node
      if (hintRuns && hintRuns.visNode <= 1) {
        setUpdateSteps(
          ".vis-node",
          "Hint",
          "Plot evaluation results (attach an evaluator or scorer node as input).",
          "left",
          true,
          false,
        );
      }
      break;
    case "uploadFileFieldsNode":
      if (hintRuns && hintRuns.uploadFileFieldsNode <= 1) {
        setUpdateSteps(
          ".file-fields-node",
          "Hint",
          "Upload a file to work with the data/context of the file. You can also declare variables in brackets { } to chain File-Fields together.",
          "left",
          true,
          false,
        );
      }
      break;
    default:
      break;
  }
};
