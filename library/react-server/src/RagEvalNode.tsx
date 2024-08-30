import React, {
  useState,
  useCallback,
  useEffect,
  useRef,
  useContext,
} from "react";
import { Handle, Position } from "reactflow";
import { NativeSelect, Flex, Text } from "@mantine/core";
import { IconScan, IconSearch } from "@tabler/icons-react";
import BaseNode from "./BaseNode";
import NodeLabel from "./NodeLabelComponent";
import InspectFooter from "./InspectFooter";
import LLMResponseInspectorModal, {
  LLMResponseInspectorModalRef,
} from "./LLMResponseInspectorModal";
import useStore from "./store";
import { ragas_evaluation, deepeval_evaluation } from "./backend/utils";
import LLMResponseInspectorDrawer from "./LLMResponseInspectorDrawer";
import { AlertModalContext } from "./AlertModal";
import { Status } from "./StatusIndicatorComponent";
import { JSONCompatible, LLMResponse, Dict } from "./backend/typing";
import TemplateHooks from "./TemplateHooksComponent";
import { v4 as uuid } from "uuid";
import { groupBy } from "lodash";

type EvaluationFormat = "Ragas" | "Deep eval";
const EVALUATION_FORMATS: EvaluationFormat[] = ["Ragas", "Deep eval"];

export interface RagEvalNodeProps {
  data: {
    vars: string[];
    evaluationFormat: EvaluationFormat;
    input: JSONCompatible[];
    refresh: boolean;
    title: string;
  };
  id: string;
}

/**
 * A no-code evaluator node with a very basic options for scoring responses.
 */
const RagEvalNode: React.FC<RagEvalNodeProps> = ({ data, id }) => {
  const setDataPropsForNode = useStore(
    (state: any) => state.setDataPropsForNode,
  );
  const pullInputData = useStore((state: any) => state.pullInputData);
  const [templateVars, setTemplateVars] = useState<string[]>([
    "questions",
    "ground_truth",
    "answers",
  ]);
  const pingOutputNodes = useStore((state: any) => state.pingOutputNodes);
  const bringNodeToFront = useStore((state: any) => state.bringNodeToFront);
  const [pastInputs, setPastInputs] = useState<JSONCompatible[]>([]);

  const [status, setStatus] = useState<Status>(Status.NONE);
  const showAlert = useContext(AlertModalContext);

  const inspectModal = useRef<LLMResponseInspectorModalRef>(null);
  // eslint-disable-next-line
  const [uninspectedResponses, setUninspectedResponses] = useState(false);
  const [lastResponses, setLastResponses] = useState<LLMResponse[]>([]);
  const [lastRunSuccess, setLastRunSuccess] = useState(true);
  const [showDrawer, setShowDrawer] = useState(false);

  const [evaluationFormat, setEvaluationFormat] = useState<EvaluationFormat>(
    data.evaluationFormat ?? "Ragas",
  );

  const [hooksY, setHooksY] = useState(102);

  const dirtyStatus = useCallback(() => {
    if (status === Status.READY) setStatus(Status.WARNING);
  }, [status]);

  const handlePullInputs = useCallback(() => {
    // Pull input data
    const pulled_inputs = pullInputData(templateVars, id);
    if (
      !pulled_inputs ||
      !pulled_inputs.questions ||
      !pulled_inputs.ground_truth ||
      !pulled_inputs.answers
    ) {
      console.warn(`Missing inputs to the Rag Evaluator node.`);
      return [];
    }
    return pulled_inputs;
  }, [pullInputData, id]);

  const getAnswerWithEval = (ansObj: any): string => {
    const ansStr =
      ansObj.answer +
      `
    ANSWER CORRECTNESS: ${ansObj.answer_correctness} `;
    if (ansObj.answer_similarity) {
      return ansStr + `ANSWER SIMILARITY: ${ansObj.answer_similarity} `;
    }
    return ansStr;
  };

  const transformToLLMResponseFormat = (
    resps: any,
    dataObj: any,
  ): LLMResponse[] =>
    resps.map((r: any) => {
      const addDataFromObj = dataObj.find((obj: any) => obj.text === r.answer);
      return {
        vars: {
          questions: r.question,
          ground_truth: r.ground_truth,
        },
        metavars: {},
        uid: addDataFromObj?.uid ? addDataFromObj.uid : uuid(),
        prompt: r.question,
        responses: [getAnswerWithEval(r)],
        tokens: addDataFromObj?.tokens ? addDataFromObj.tokens : {},
        llm: addDataFromObj?.llm ? addDataFromObj.llm : "undefined",
      };
    });

  const handleRunClick = useCallback(async () => {
    // Pull inputs to the node
    const pulled_inputs = handlePullInputs();
    let responses: any = [];

    const llm_answers: any = groupBy(
      pulled_inputs.answers as any[],
      ({ llm }: { llm: { name: string } }) => llm.name,
    );
    // Set status and created rejection callback
    setStatus(Status.LOADING);
    setLastResponses([]);

    const rejected = (err_msg: string) => {
      setStatus(Status.ERROR);
      setLastRunSuccess(false);
      if (showAlert) showAlert(err_msg);
    };

    for (const element of Object.keys(llm_answers)) {
      const params: {
        question: string[];
        ground_truth: string[];
        answer: string[];
      } = {
        question: [],
        ground_truth: [],
        answer: [],
      };
      pulled_inputs.questions.forEach((question: any) => {
        params.question.push(question.text);
        const ground_truth = pulled_inputs.ground_truth.find((obj: any) =>
          JSON.stringify(obj.metavars).includes(question.text),
        );
        if (ground_truth) params.ground_truth.push(ground_truth.text);
        const ans = llm_answers[element].find(
          (obj: any) => obj.prompt === question.text,
        );
        if (ans) params.answer.push(ans.text);
      });
      try {
        let resps: any;
        if (evaluationFormat === "Deep eval") {
          resps = await deepeval_evaluation(params);
        } else {
          resps = await ragas_evaluation(params);
        }
        // Check if there's an error; if so, bubble it up to user and exit:
        if (resps.error || resps === undefined) throw new Error(resps.error);
        // Ping any vis + inspect nodes attached to this node to refresh their contents:
        responses = [
          ...responses,
          ...transformToLLMResponseFormat(resps, llm_answers[element]),
        ];
      } catch (err: any) {
        rejected(typeof err === "string" ? err : err.message);
      }
    }

    pingOutputNodes(id);
    setLastResponses(responses);
    setLastRunSuccess(true);

    if (status !== Status.READY && !showDrawer) setUninspectedResponses(true);

    setStatus(Status.READY);
  }, [
    handlePullInputs,
    pingOutputNodes,
    setStatus,
    showAlert,
    status,
    evaluationFormat,
    showDrawer,
  ]);

  const showResponseInspector = useCallback(() => {
    if (inspectModal && inspectModal.current && lastResponses) {
      setUninspectedResponses(false);
      inspectModal.current.trigger();
    }
  }, [inspectModal, lastResponses]);

  if (data.input) {
    // If there's a change in inputs...
    if (data.input !== pastInputs) {
      setPastInputs(data.input);
      handlePullInputs();
    }
  }

  useEffect(() => {
    if (data.refresh && data.refresh === true) {
      setDataPropsForNode(id, { refresh: false });
      setStatus(Status.WARNING);
      handlePullInputs();
    }
  }, [data]);

  return (
    <BaseNode classNames="evaluator-node" nodeId={id}>
      <NodeLabel
        title={data.title || "Rag Evaluator"}
        nodeId={id}
        icon={<IconScan size="16px" />}
        status={status}
        handleRunClick={handleRunClick}
        runButtonTooltip="Run evaluator over inputs"
      />

      <LLMResponseInspectorModal
        ref={inspectModal}
        jsonResponses={lastResponses}
      />
      <iframe style={{ display: "none" }} id={`${id}-iframe`}></iframe>

      <Flex gap="xs">
        <Text mt="6px" fz="sm">
          Evaluator
        </Text>
        <NativeSelect
          data={EVALUATION_FORMATS}
          defaultValue={evaluationFormat}
          onChange={(e) => {
            setEvaluationFormat(e.target.value as EvaluationFormat);
            setDataPropsForNode(id, {
              evaluationFormat: e.target.value as EvaluationFormat,
            });
            dirtyStatus();
          }}
        />
      </Flex>

      <TemplateHooks
        vars={templateVars}
        nodeId={id}
        startY={hooksY}
        position={Position.Left}
      />
      <Handle
        type="source"
        position={Position.Right}
        id="output"
        className="grouped-handle"
        style={{ top: "50%" }}
      />

      {lastRunSuccess && lastResponses && lastResponses.length > 0 ? (
        <InspectFooter
          label={
            <>
              Inspect scores&nbsp;
              <IconSearch size="12pt" />
            </>
          }
          onClick={showResponseInspector}
          isDrawerOpen={showDrawer}
          showDrawerButton={true}
          onDrawerClick={() => {
            setShowDrawer(!showDrawer);
            setUninspectedResponses(false);
            bringNodeToFront(id);
          }}
        />
      ) : (
        <></>
      )}

      <LLMResponseInspectorDrawer
        jsonResponses={lastResponses}
        showDrawer={showDrawer}
      />
    </BaseNode>
  );
};

export default RagEvalNode;
