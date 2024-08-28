import React, {
  useState,
  useCallback,
  useRef,
  useEffect,
  useContext,
} from "react";
import ReactFlow, {
  Controls,
  Background,
  ReactFlowInstance,
  Node,
  MiniMap,
} from "reactflow";
import {
  Button,
  Menu,
  LoadingOverlay,
  Text,
  Box,
  List,
  Loader,
  Tooltip,
  Modal,
  Textarea,
  TextInput,
  Input,
  Flex,
} from "@mantine/core";
import { useClipboard } from "@mantine/hooks";
import { useContextMenu } from "mantine-contextmenu";
import {
  IconSettings,
  IconTextPlus,
  IconTerminal,
  IconSettingsAutomation,
  IconFileSymlink,
  IconRobot,
  IconRuler2,
  IconArrowMerge,
  IconArrowsSplit,
  IconForms,
  IconAbacus,
  IconCheck,
  IconX,
} from "@tabler/icons-react";
import { useNotification } from "./Notification";
import RemoveEdge from "./RemoveEdge";
import TextFieldsNode from "./TextFieldsNode"; // Import a custom node
import UploadFileFieldsNode from "./UploadFileFieldsNode"; // Import a custom node
import PromptNode from "./PromptNode";
import CodeEvaluatorNode from "./CodeEvaluatorNode";
import VisNode from "./VisNode";
import InspectNode from "./InspectorNode";
import ScriptNode from "./ScriptNode";
import { AlertModalContext } from "./AlertModal";
import ItemsNode from "./ItemsNode";
import TabularDataNode from "./TabularDataNode";
import JoinNode from "./JoinNode";
import SplitNode from "./SplitNode";
import CommentNode from "./CommentNode";
import GlobalSettingsModal, {
  GlobalSettingsModalRef,
} from "./GlobalSettingsModal";
import ExampleFlowsModal, { ExampleFlowsModalRef } from "./ExampleFlowsModal";
import AreYouSureModal, { AreYouSureModalRef } from "./AreYouSureModal";
import LLMEvaluatorNode from "./LLMEvalNode";
import SimpleEvalNode from "./SimpleEvalNode";
import {
  getDefaultModelFormData,
  getDefaultModelSettings,
} from "./ModelSettingSchemas";
import { v4 as uuid } from "uuid";
import LZString from "lz-string";
import { EXAMPLEFLOW_1 } from "./example_flows";

// Styling
import "reactflow/dist/style.css"; // reactflow
import "./text-fields-node.css"; // project

// Lazy loading images
import "lazysizes";
import "lazysizes/plugins/attrchange/ls.attrchange";

// State management (from https://reactflow.dev/docs/guides/state-management/)
import { shallow } from "zustand/shallow";
import useStore, { StoreHandles } from "./store";
import StorageCache from "./backend/cache";
import {
  FLASK_BASE_URL,
  APP_IS_RUNNING_LOCALLY,
  browserTabIsActive,
} from "./backend/utils";
import { Dict, JSONCompatible, LLMSpec } from "./backend/typing";
import {
  exportCache,
  fetchEnvironAPIKeys,
  fetchExampleFlow,
  fetchOpenAIEval,
  importCache,
} from "./backend/backend";

// Device / Browser detection
import {
  isMobile,
  isChrome,
  isFirefox,
  isEdgeChromium,
  isChromium,
} from "react-device-detect";
import MultiEvalNode from "./MultiEvalNode";
import { BinIcon, Chevron, CopyIcon, LockIcon, TickMark } from "./SvgIcons";
import "./CssStyles.css";
const IS_ACCEPTED_BROWSER =
  (isChrome ||
    isChromium ||
    isEdgeChromium ||
    isFirefox ||
    (navigator as any)?.brave !== undefined) &&
  !isMobile;

// Whether we are running on localhost or not, and hence whether
// we have access to the Flask backend for, e.g., Python code evaluation.
const IS_RUNNING_LOCALLY = APP_IS_RUNNING_LOCALLY();

const selector = (state: StoreHandles) => ({
  nodes: state.nodes,
  edges: state.edges,
  onNodesChange: state.onNodesChange,
  onEdgesChange: state.onEdgesChange,
  onConnect: state.onConnect,
  addNode: state.addNode,
  setNodes: state.setNodes,
  setEdges: state.setEdges,
  resetLLMColors: state.resetLLMColors,
  setAPIKeys: state.setAPIKeys,
  importState: state.importState,
});

// The initial LLM to use when new flows are created, or upon first load
const INITIAL_LLM = () => {
  if (!IS_RUNNING_LOCALLY) {
    // Prefer HF if running on server, as it's free.
    const falcon7b = {
      key: uuid(),
      name: "Mistral-7B",
      emoji: "🤗",
      model: "mistralai/Mistral-7B-Instruct-v0.1",
      base_model: "hf",
      temp: 1.0,
      settings: getDefaultModelSettings("hf"),
      formData: getDefaultModelFormData("hf"),
    } satisfies LLMSpec;
    falcon7b.formData.shortname = falcon7b.name;
    falcon7b.formData.model = falcon7b.model;
    return falcon7b;
  } else {
    // Prefer OpenAI for majority of local users.
    const chatgpt = {
      key: uuid(),
      name: "GPT3.5",
      emoji: "🤖",
      model: "gpt-3.5-turbo",
      base_model: "gpt-3.5-turbo",
      temp: 1.0,
      settings: getDefaultModelSettings("gpt-3.5-turbo"),
      formData: getDefaultModelFormData("gpt-3.5-turbo"),
    } satisfies LLMSpec;
    chatgpt.formData.shortname = chatgpt.name;
    chatgpt.formData.model = chatgpt.model;
    return chatgpt;
  }
};

const nodeTypes = {
  textfields: TextFieldsNode, // Register the custom node
  uploadfilefields: UploadFileFieldsNode, // Register the custom node
  prompt: PromptNode,
  chat: PromptNode,
  simpleval: SimpleEvalNode,
  evaluator: CodeEvaluatorNode,
  llmeval: LLMEvaluatorNode,
  multieval: MultiEvalNode,
  vis: VisNode,
  inspect: InspectNode,
  script: ScriptNode,
  csv: ItemsNode,
  table: TabularDataNode,
  comment: CommentNode,
  join: JoinNode,
  split: SplitNode,
  processor: CodeEvaluatorNode,
};

const edgeTypes = {
  default: RemoveEdge,
};

// Try to get a GET param in the URL, representing the shared flow.
// Returns undefined if not found.
const getSharedFlowURLParam = () => {
  // Get the current URL
  const curr_url = new URL(window.location.href);

  // Get the search parameters from the URL
  const params = new URLSearchParams(curr_url.search);

  // Try to retrieve an 'f' parameter (short for flow)
  const shared_flow_uid = params.get("f");

  if (shared_flow_uid) {
    // Check if it's a base36 string:
    const is_base36 = /^[0-9a-z]+$/i;
    if (shared_flow_uid.length > 1 && is_base36.test(shared_flow_uid))
      return shared_flow_uid;
  }
  return undefined;
};

const MenuTooltip = ({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) => {
  return (
    <Tooltip
      label={label}
      position="right"
      width={200}
      multiline
      withArrow
      arrowSize={10}
    >
      {children}
    </Tooltip>
  );
};

// const connectionLineStyle = { stroke: '#ddd' };
const snapGrid: [number, number] = [16, 16];

const App = () => {
  // Get nodes, edges, etc. state from the Zustand store:
  const {
    nodes,
    edges,
    onNodesChange,
    onEdgesChange,
    onConnect,
    addNode: addNodeToStore,
    setNodes,
    setEdges,
    resetLLMColors,
    setAPIKeys,
    importState,
  } = useStore(selector, shallow);

  const { showNotification } = useNotification();
  const [isUseCaseCreated, setIsUseCaseCreated] = useState<any>();
  const [openCreateUseCase, setOpenCreateUseCase] = useState(false);
  const [useCaseName, setUseCaseName] = useState("");
  const [description, setDescription] = useState("");
  const [errorMessage, setErrorMessage] = useState({
    error: false,
    message: "",
  });
  const [loading, setLoading] = useState(false);
  const [menuData, setMenuData] = useState<any>(null);
  const [openMenu, setOpenMenu] = useState<any>(false);
  const [openAddNode, setOpenAddNode] = useState(false);
  const [activeUseCase, setActiveUseCase] = useState({
    usecase: "",
    iteration: "",
    fileName: "",
    committed: false,
  });
  const [deleteusecaseOrIter, setDeleteUsecaseOrIter] = useState({
    usecase: "",
    iteration: "",
    open: false,
  });

  const [saveDropdown, setSaveDropdown] = useState("Save");
  const [isCurrenFileLocked, setIsCurrentFileLocked] = useState(false);
  const [saveAndCommitBtnOpen, setSaveAndCommitBtnOpen] = useState(false);
  const [isChangesNotSaved, setIsChangesNotSaved] = useState(false);
  const [confirmed, setConfirmed] = useState(false);
  const [modalOpen, setModalOpen] = useState<{
    usecase: string;
    iteration: string;
    subItems: any[];
    open: boolean;
    for: string;
  }>({
    usecase: "",
    iteration: "",
    subItems: [],
    open: false,
    for: "",
  });
  const [copyModalOpen, setCopyModalOpen] = useState({
    usecase: "",
    open: false,
    for: "",
  });
  const [warning, setWarning] = useState({ warning: "", open: false });
  const API_URL = process.env.REACT_APP_API_URL;
  const [hoveredItem, setHoveredItem] = useState(null);
  const [editUsecaseforCopy, setEditUsecaseforCopy] = useState("");
  // For saving / loading
  const [rfInstance, setRfInstance] = useState<ReactFlowInstance | null>(null);
  const [autosavingInterval, setAutosavingInterval] = useState<
    NodeJS.Timeout | undefined
  >(undefined);
  // For 'share' button
  const clipboard = useClipboard({ timeout: 1500 });
  const [waitingForShare, setWaitingForShare] = useState(false);

  // For modal popup to set global settings like API keys
  const settingsModal = useRef<GlobalSettingsModalRef>(null);
  const saveRef = useRef<any>(null);
  // For modal popup of example flows
  const examplesModal = useRef<ExampleFlowsModalRef>(null);
  const queryString = window.location.search;
  const urlParams = new URLSearchParams(queryString);

  // For an info pop-up that welcomes new users
  // const [welcomeModalOpened, { open: openWelcomeModal, close: closeWelcomeModal }] = useDisclosure(false);

  // For displaying alerts
  const showAlert = useContext(AlertModalContext);

  // For confirmation popup
  const confirmationModal = useRef<AreYouSureModalRef>(null);
  const [confirmationDialogProps, setConfirmationDialogProps] = useState<{
    title: string;
    message: string;
    onConfirm?: () => void;
  }>({
    title: "Confirm action",
    message: "Are you sure?",
  });

  // For Mantine Context Menu forced closing
  // (for some reason the menu doesn't close automatically upon click-off)
  const { hideContextMenu } = useContextMenu();

  // For displaying a pending 'loading' status
  const [isLoading, setIsLoading] = useState(true);

  // Helper
  const getWindowSize = () => ({
    width: window.innerWidth,
    height: window.innerHeight,
  });
  const getWindowCenter = () => {
    const { width, height } = getWindowSize();
    return { centerX: width / 2.0, centerY: height / 2.0 };
  };
  const getViewportCenter = () => {
    const { centerX, centerY } = getWindowCenter();
    if (rfInstance === null) return { x: centerX, y: centerY };
    // Support Zoom
    const { x, y, zoom } = rfInstance.getViewport();
    return { x: -(x / zoom) + centerX / zoom, y: -(y / zoom) + centerY / zoom };
  };

  const addNode = (
    id: string,
    type?: string,
    data?: Dict,
    offsetX?: number,
    offsetY?: number,
  ) => {
    const { x, y } = getViewportCenter();
    addNodeToStore({
      id: `${id}-` + Date.now(),
      type: type ?? id,
      data: data ?? {},
      position: {
        x: x - 200 + (offsetX || 0),
        y: y - 100 + (offsetY || 0),
      },
    });
    setIsChangesNotSaved(true);
    // following changes are for showing warning if we delete iter or usecase and again if we try to add nodes and save flow
    const data1: any = localStorage.getItem("current_usecase");
    const localData = JSON.parse(data1);
    if (localData != null) {
      if (!localData.file_name && !localData.iter_folder && warning.open) {
        showNotification(
          "Failed",
          "Please select use case and iteration, otherwise changes will be lost",
          "red",
        );
        setWarning({
          warning: "",
          open: false,
        });
      }
    }
    setOpenAddNode(false);
  };

  const addTextFieldsNode = () => addNode("textFieldsNode", "textfields");
  const addUploadFileFieldsNode = () =>
    addNode("uploadFileFieldsNode", "uploadfilefields");
  const addPromptNode = () => addNode("promptNode", "prompt", { prompt: "" });
  const addChatTurnNode = () => addNode("chatTurn", "chat", { prompt: "" });
  const addSimpleEvalNode = () => addNode("simpleEval", "simpleval");
  const addEvalNode = (progLang: string) => {
    let code = "";
    if (progLang === "python")
      code = "def evaluate(response):\n  return len(response.text)";
    else if (progLang === "javascript")
      code = "function evaluate(response) {\n  return response.text.length;\n}";
    addNode("evalNode", "evaluator", { language: progLang, code });
  };
  const addVisNode = () => addNode("visNode", "vis", {});
  const addInspectNode = () => addNode("inspectNode", "inspect");
  const addScriptNode = () => addNode("scriptNode", "script");
  const addItemsNode = () => addNode("csvNode", "csv");
  const addTabularDataNode = () => addNode("table");
  const addCommentNode = () => addNode("comment");
  const addLLMEvalNode = () => addNode("llmeval");
  const addMultiEvalNode = () => addNode("multieval");
  const addJoinNode = () => addNode("join");
  const addSplitNode = () => addNode("split");
  const addProcessorNode = (progLang: string) => {
    let code = "";
    if (progLang === "python")
      code = "def process(response):\n  return response.text;";
    else if (progLang === "javascript")
      code = "function process(response) {\n  return response.text;\n}";
    addNode("process", "processor", { language: progLang, code });
  };

  const onClickExamples = () => {
    if (examplesModal && examplesModal.current) examplesModal.current.trigger();
  };
  const onClickSettings = () => {
    if (settingsModal && settingsModal.current) settingsModal.current.trigger();
  };

  const handleError = (err: Error | string) => {
    const msg = typeof err === "string" ? err : err.message;
    setIsLoading(false);
    setWaitingForShare(false);
    if (showAlert) showAlert(msg);
    console.error(msg);
  };

  /**
   * SAVING / LOADING, IMPORT / EXPORT (from JSON)
   */
  const downloadJSON = (jsonData: JSONCompatible, filename: string) => {
    // Convert JSON object to JSON string
    const jsonString = JSON.stringify(jsonData, null, 2);

    // Create a Blob object from the JSON string
    const blob = new Blob([jsonString], { type: "application/json" });

    // Create a temporary download link
    const downloadLink = document.createElement("a");
    downloadLink.href = URL.createObjectURL(blob);
    downloadLink.download = filename;

    // Add the link to the DOM (it's not visible)
    document.body.appendChild(downloadLink);

    // Trigger the download by programmatically clicking the temporary link
    downloadLink.click();

    // Remove the temporary link from the DOM and revoke the URL
    document.body.removeChild(downloadLink);
    URL.revokeObjectURL(downloadLink.href);
  };

  // Save the current flow to localStorage for later recall. Useful to getting
  // back progress upon leaving the site / browser crash / system restart.
  const saveFlow = useCallback(
    (rf_inst: ReactFlowInstance) => {
      const rf = rf_inst ?? rfInstance;
      if (!rf) return;

      // NOTE: This currently only saves the front-end state. Cache files
      // are not pulled or overwritten upon loading from localStorage.
      const flow = rf.toObject();
      StorageCache.saveToLocalStorage("aggrag-flow", flow);

      // Attempt to save the current state of the back-end state,
      // the StorageCache. (This does LZ compression to save space.)
      StorageCache.saveToLocalStorage("aggrag-state");

      console.log("Flow saved!");
    },
    [rfInstance],
  );

  // Triggered when user confirms 'New Flow' button
  const resetFlow = useCallback(() => {
    resetLLMColors();

    const uid = (id: string) => `${id}-${Date.now()}`;
    const starting_nodes = [
      {
        id: uid("prompt"),
        type: "prompt",
        data: {
          prompt: "",
          n: 1,
          llms: [INITIAL_LLM()],
        },
        position: { x: 450, y: 200 },
      },
      {
        id: uid("textfields"),
        type: "textfields",
        data: {},
        position: { x: 80, y: 270 },
      },
    ];

    setNodes(starting_nodes);
    setEdges([]);
    if (rfInstance) rfInstance.setViewport({ x: 200, y: 80, zoom: 1 });
  }, [setNodes, setEdges, resetLLMColors, rfInstance]);

  const resetFlowToBlankCanvas = useCallback(() => {
    resetLLMColors();

    const uid = (id: string) => `${id}-${Date.now()}`;
    const starting_nodes: Node[] = [];

    setNodes(starting_nodes);
    setEdges([]);
    if (rfInstance) rfInstance.setViewport({ x: 200, y: 80, zoom: 1 });
  }, [setNodes, setEdges, resetLLMColors, rfInstance]);

  const loadFlow = async (flow?: Dict, rf_inst?: ReactFlowInstance | null) => {
    if (flow === undefined) return;
    if (rf_inst) {
      if (flow.viewport) {
        if (rf_inst && flow.nodes.length > 10) {
          rf_inst.setViewport({ x: 0, y: 0, zoom: 0.3 });
        } else
          rf_inst.setViewport({
            x: flow.viewport.x || 0,
            y: flow.viewport.y || 0,
            // zoom: flow.viewport.zoom || 1,
            zoom: flow.viewport.zoom || 1,
          });
      } else rf_inst.setViewport({ x: 0, y: 0, zoom: 1 });
    }
    resetLLMColors();

    // First, clear the ReactFlow state entirely
    // NOTE: We need to do this so it forgets any node/edge ids, which might have cross-over in the loaded flow.
    setNodes([]);
    setEdges([]);

    // After a delay, load in the new state.
    setTimeout(() => {
      setNodes(flow.nodes || []);
      setEdges(flow.edges || []);

      // Save flow that user loaded to autosave cache, in case they refresh the browser
      StorageCache.saveToLocalStorage("aggrag-flow", flow);

      // Cancel loading spinner
      setIsLoading(false);
    }, 10);

    // Start auto-saving, if it's not already enabled
    // if (rf_inst) initAutosaving(rf_inst);
  };

  const importGlobalStateFromCache = useCallback(() => {
    importState(StorageCache.getAllMatching((key) => key.startsWith("r.")));
  }, [importState]);

  const autosavedFlowExists = () => {
    return window.localStorage.getItem("aggrag-flow") !== null;
  };
  const loadFlowFromAutosave = async (rf_inst: ReactFlowInstance) => {
    const saved_flow = StorageCache.loadFromLocalStorage(
      "aggrag-flow",
      false,
    ) as Dict;
    if (saved_flow) {
      StorageCache.loadFromLocalStorage("aggrag-state");
      importGlobalStateFromCache();
      loadFlow(saved_flow, rf_inst);
    }
  };

  // Export / Import (from JSON)
  const exportFlow = useCallback(async () => {
    const data: any = {};
    data.folder_path = `configurations/${urlParams.get("p_folder")}/${urlParams.get("i_folder")}`;
    fetch(
      `${FLASK_BASE_URL}app/exportFiles?` +
        new URLSearchParams(data).toString(),
      {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
          "Access-Control-Allow-Origin": "*",
        },
      },
    )
      .then((res) => {
        return res.blob();
      })
      .then((blob) => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `${urlParams.get("i_folder")}.zip`;
        document.body.appendChild(a);
        a.click();
        a.remove();
      })
      .catch(handleError);
  }, [handleError]);

  const handleSaveAndCommit = () => {
    handleSaveFlow(true);
  };

  const handleSaveFlow = (
    saveAndCommit: boolean,
    forIterationCreation?: boolean,
  ) => {
    setLoading(true);
    setOpenMenu(false);
    const iterationCreation = forIterationCreation ?? false;
    if (menuData && menuData.length === 0) {
      setLoading(false);
      return;
    }
    localStorage.setItem(
      "iteration-created",
      JSON.stringify({
        usecase: "",
        iteration: "",
        fileName: "",
        committed: false,
        iterationCreated: false,
      }),
    );
    setIsChangesNotSaved(false);
    if (!rfInstance) return;

    // We first get the data of the flow
    const flow = rfInstance.toObject();
    const newFlow = {
      ...flow,
      isCommitted: saveAndCommit,
    };
    // Then we grab all the relevant cache files from the backend
    const all_node_ids = nodes.map((n) => n.id);
    let localStorageData: any = localStorage.getItem("current_usecase");
    if (localStorageData === null) {
      localStorage.setItem(
        "current_usecase",
        JSON.stringify({
          parent_folder: "",
          iter_folder: "",
          file_name: "",
          committed: false,
        }),
      );
      localStorageData = localStorage.getItem("current_usecase");
    }

    const currUseCase = JSON.parse(localStorageData);
    exportCache(all_node_ids).then(function (cacheData) {
      const requestData = {
        flow: saveAndCommit ? newFlow : flow,
        cache: cacheData,
        folderName: currUseCase.parent_folder,
        iterationName: currUseCase.iter_folder,
        timestamp: Date.now(),
        fileName: currUseCase.file_name,
      };
      fetch(`${API_URL}app/saveflow`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestData),
      })
        .then((response) => {
          if (!response.ok) {
            showNotification(
              "Failed",
              "File not saved, please select one of the usecase(path)",
              "red",
            );
            throw new Error("Failed to save flow.");
          }
          return response.json();
        })
        .then((data) => {
          if (!isUseCaseCreated) {
            setIsUseCaseCreated(true);
          } else {
            setIsUseCaseCreated(false);
          }
          const temp_file_name = saveAndCommit ? "" : data.file_name;
          if (saveAndCommit) {
            setIsCurrentFileLocked(true);
            setSaveDropdown("Deploy in app");
          }
          !saveAndCommit &&
            localStorage.setItem(
              "current_usecase",
              JSON.stringify({
                parent_folder: currUseCase.parent_folder,
                iter_folder: currUseCase.iter_folder,
                file_name: temp_file_name,
                committec: false,
              }),
            );
          // here also we can put committed value as false ( already save and commit is false)
          !saveAndCommit &&
            setActiveUseCase({
              usecase: currUseCase.parent_folder,
              iteration: currUseCase.iter_folder,
              fileName: temp_file_name,
              committed: false,
            });

          const queryString = `?p_folder=${encodeURIComponent(currUseCase.parent_folder)}&i_folder=${encodeURIComponent(currUseCase.iter_folder)}&file_name=${encodeURIComponent(temp_file_name)}`;
          if (saveAndCommit) {
            showNotification(
              "Saved & Committed",
              "File has been successfully saved & committed!",
            );
          } else if (!iterationCreation) {
            showNotification(
              "Saved",
              `File has been successfully saved (${currUseCase.iter_folder} of ${currUseCase && currUseCase.parent_folder.split("__")[0]})`,
            );
          }
          // Update the URL
          !saveAndCommit && window.history.pushState({}, "", queryString);
          setLoading(false);
        })
        .catch((error) => {
          setLoading(false);
          console.error("Error saving flow:", error);
          // setNotificationText({
          //   title: "Failed",
          //   text: "File not saved, please select one of the usecase(path)",
          // });
        });
    });
  };
  const handleIterationFolderClick = (
    ItemLabel: any,
    subItemLabel: any,
    subItems: any,
  ) => {
    try {
      setOpenMenu(false);
      if (isChangesNotSaved) {
        if (
          activeUseCase.usecase === ItemLabel &&
          activeUseCase.iteration === subItemLabel
        ) {
          return;
        } else {
          setModalOpen({
            usecase: ItemLabel,
            iteration: subItemLabel,
            subItems: subItems,
            open: true,
            for: "click-iteration",
          });
        }

        return;
      } else {
        setModalOpen({
          usecase: "",
          iteration: "",
          subItems: [],
          open: false,
          for: "",
        });
      }
      let currentFile = "";

      if (subItems && subItems.length > 0) {
        // here we use this subItems[0]?.isCommitted to store committed value in active use case
        currentFile = subItems[0]?.label;
        try {
          loadExistingFile(
            ItemLabel,
            subItemLabel,
            currentFile,
            subItems[0]?.isCommitted,
          );
        } catch (e) {
          console.log("error in loading file");
        }
        if (subItems[0]?.isCommitted) {
          setSaveDropdown("Deploy in app");
        } else {
          setSaveDropdown("Save");
        }

        setActiveUseCase({
          usecase: ItemLabel,
          iteration: subItemLabel,
          fileName: currentFile,
          committed: subItems[0]?.isCommitted,
        });
        localStorage.setItem(
          "current_usecase",
          JSON.stringify({
            parent_folder: ItemLabel,
            iter_folder: subItemLabel,
            file_name: currentFile,
            committed: subItems[0]?.isCommitted,
          }),
        );
      } else {
        resetFlowToBlankCanvas();
        setIsCurrentFileLocked(false);
        setSaveDropdown("Save");
        localStorage.setItem(
          "current_usecase",
          JSON.stringify({
            parent_folder: ItemLabel,
            iter_folder: subItemLabel,
            file_name: currentFile,
            committed: false,
          }),
        );
        // here we can declare is committed value as false
        setActiveUseCase({
          usecase: ItemLabel,
          iteration: subItemLabel,
          fileName: currentFile,
          committed: false,
        });
        const queryString = `?p_folder=${encodeURIComponent(ItemLabel)}&i_folder=${encodeURIComponent(subItemLabel)}&file_name=${encodeURIComponent(currentFile)}`;
        // Update the URL
        window.history.pushState({}, "", queryString);
      }
    } catch (e) {
      console.log("error in iteration and file retrieving");
    }
  };
  const loadExistingFile = async (
    parent_folder: string,
    iter_folder: string,
    file_name: string,
    committed: boolean,
  ) => {
    localStorage.setItem(
      "current_usecase",
      JSON.stringify({
        parent_folder,
        iter_folder,
        file_name,
        committed,
      }),
    );
    const queryString = `?p_folder=${encodeURIComponent(parent_folder)}&i_folder=${encodeURIComponent(iter_folder)}&file_name=${encodeURIComponent(file_name)}`;

    // Update the URL
    window.history.pushState({}, "", queryString);

    setOpenMenu(false);
    const relativeFilePath = `${parent_folder}/${iter_folder}/${file_name}`;
    try {
      const response = await fetch(
        `${API_URL}app/getfile?file_path=${encodeURIComponent(relativeFilePath)}`,
      );
      if (!response.ok) {
        throw new Error("Failed to fetch the file.");
      }
      const flow_and_cache = await response.json();
      importFlowFromJSON(flow_and_cache);
      // pending ( may be no need to save active use case her)
      setActiveUseCase({
        usecase: parent_folder,
        iteration: iter_folder,
        fileName: file_name,
        committed: committed,
      });
    } catch (error: any) {
      // handleError(error);
    }
  };

  // Import data to the cache stored on the local filesystem (in backend)
  const handleImportCache = useCallback(
    (cache_data: Dict<Dict>) =>
      importCache(cache_data)
        .then(importGlobalStateFromCache)
        .catch(handleError),
    [handleError, importGlobalStateFromCache],
  );

  const importFlowFromJSON = useCallback(
    (flowJSON: Dict, rf_inst?: ReactFlowInstance | null) => {
      const rf = rf_inst ?? rfInstance;

      setIsLoading(true);

      // Detect if there's no cache data
      if (!flowJSON.cache) {
        // Support for loading old flows w/o cache data:
        loadFlow(flowJSON, rf);
        return;
      }

      // Then we need to extract the JSON of the flow vs the cache data
      const flow = flowJSON.flow;
      if (flow.isCommitted) {
        setIsCurrentFileLocked(true);
        setSaveDropdown("Deploy in app");
      } else {
        setIsCurrentFileLocked(false);
      }
      const cache = flowJSON.cache;
      // We need to send the cache data to the backend first,
      // before we can load the flow itself...
      handleImportCache(cache)
        .then(() => {
          // We load the ReactFlow instance last
          loadFlow(flow, rf);
        })
        .catch((err) => {
          // On an error, still try to load the flow itself:
          handleError(
            "Error encountered when importing cache data:" +
              err.message +
              "\n\nTrying to load flow regardless...",
          );
          loadFlow(flow, rf);
        });
    },
    [rfInstance],
  );

  // Import a Aggrag flow from a file
  const importFlowFromFile = async () => {
    // Create an input element with type "file" and accept only JSON files
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".cforge, .json";

    // Handle file selection
    input.addEventListener(
      "change",
      // @ts-expect-error The event is correctly typed here, but for some reason TS doesn't pick up on it.
      function (event: React.ChangeEvent<HTMLInputElement>) {
        // Start loading spinner
        setIsLoading(false);

        const files = event.target.files;
        if (!files || typeof files !== "object" || files.length === 0) {
          console.error("No files found to load.");
          return;
        }

        const file = files[0];
        const reader = new window.FileReader();

        // Handle file load event
        reader.addEventListener("load", function () {
          try {
            if (typeof reader.result !== "string")
              throw new Error(
                "File could not be read: Unknown format or empty.",
              );

            // We try to parse the JSON response
            const flow_and_cache = JSON.parse(reader.result);

            // Import it to React Flow and import cache data on the backend
            importFlowFromJSON(flow_and_cache);
          } catch (error) {
            handleError(error as Error);
          }
        });

        // Read the selected file as text
        reader.readAsText(file);
      },
    );

    // Trigger the file selector
    input.click();
  };

  // Downloads the selected OpenAI eval file (preconverted to a .cforge flow)
  const importFlowFromOpenAIEval = (evalname: string) => {
    setIsLoading(true);

    fetchOpenAIEval(evalname).then(importFlowFromJSON).catch(handleError);
  };

  // Load flow from examples modal
  const onSelectExampleFlow = (name: string, example_category?: string) => {
    // Trigger the 'loading' modal
    setIsLoading(true);

    // Detect a special category of the example flow, and use the right loader for it:
    if (example_category === "openai-eval") {
      importFlowFromOpenAIEval(name);
      return;
    }

    // Fetch the example flow data from the backend
    fetchExampleFlow(name)
      .then(function (flowJSON) {
        // We have the data, import it:
        importFlowFromJSON(flowJSON);
      })
      .catch(handleError);
  };

  // When the user clicks the 'New Flow' button
  const onClickNewFlow = useCallback(() => {
    setConfirmationDialogProps({
      title: "Create a new flow",
      message:
        "Are you sure? Any unexported changes to your existing flow will be lost.",
      onConfirm: () => resetFlow(), // Set the callback if user confirms action
    });

    // Trigger the 'are you sure' modal:
    if (confirmationModal && confirmationModal.current)
      confirmationModal.current?.trigger();
  }, [confirmationModal, resetFlow, setConfirmationDialogProps]);

  // When the user clicks the 'Share Flow' button
  const onClickShareFlow = useCallback(async () => {
    if (IS_RUNNING_LOCALLY) {
      handleError(
        new Error(
          "Cannot upload flow to server database when running locally: Feature only exists on hosted version of Aggrag.",
        ),
      );
      return;
    } else if (waitingForShare === true) {
      handleError(
        new Error(
          "A share request is already in progress. Wait until the current share finishes before clicking again.",
        ),
      );
      return;
    }

    // Helper function
    function isFileSizeLessThan5MB(json_str: string) {
      const encoder = new TextEncoder();
      const encodedString = encoder.encode(json_str);
      const fileSizeInBytes = encodedString.length;
      const fileSizeInMB = fileSizeInBytes / (1024 * 1024); // Convert bytes to megabytes
      return fileSizeInMB < 5;
    }

    setWaitingForShare(true);

    // Package up the current flow:
    const flow = rfInstance?.toObject();
    const all_node_ids = nodes.map((n) => n.id);
    const cforge_data = await exportCache(all_node_ids)
      .then(function (cacheData) {
        // Now we append the cache file data to the flow
        return {
          flow,
          cache: cacheData,
        };
      })
      .catch(handleError);

    if (!cforge_data) return;

    // Compress the data and check it's compressed size < 5MB:
    const compressed = LZString.compressToUTF16(JSON.stringify(cforge_data));
    if (!isFileSizeLessThan5MB(compressed)) {
      handleError(
        new Error(
          "Flow filesize exceeds 5MB. You can only share flows up to 5MB or less. But, don't despair! You can still use 'Export Flow' to share your flow manually as a .cforge file.",
        ),
      );
      return;
    }

    // Try to upload the compressed cforge data to the server:
    fetch("/db/shareflow.php", {
      method: "POST",
      body: compressed,
    })
      .then((r) => r.text())
      .then((uid) => {
        if (!uid) {
          throw new Error("Received no response from server.");
        } else if (uid.startsWith("Error")) {
          // Error encountered during the query; alert the user
          // with the error message:
          throw new Error(uid);
        }

        // Share completed!
        setWaitingForShare(false);

        // The response should be a uid we can put in a GET request.
        // Generate the link:
        const base_url = new URL(
          window.location.origin + window.location.pathname,
        ); // the current address
        const get_params = new URLSearchParams(base_url.search);
        // Add the 'f' parameter
        get_params.set("f", uid); // set f=uid
        // Update the URL with the modified search parameters
        base_url.search = get_params.toString();
        // Get the modified URL
        const get_url = base_url.toString();

        // Copies the GET URL to user's clipboard
        // and updates the 'Share This' button state:
        clipboard.copy(get_url);
      })
      .catch((err) => {
        handleError(err);
      });
  }, [
    rfInstance,
    nodes,
    IS_RUNNING_LOCALLY,
    handleError,
    clipboard,
    waitingForShare,
  ]);

  // Initialize auto-saving
  const initAutosaving = (rf_inst: ReactFlowInstance) => {
    if (autosavingInterval !== undefined) return; // autosaving interval already set
    console.log("Init autosaving");

    // Autosave the flow to localStorage every minute:
    const interv = setInterval(() => {
      // Check the visibility of the browser tab --if it's not visible, don't autosave
      if (!browserTabIsActive()) return;

      // Start a timer, in case the saving takes a long time
      const startTime = Date.now();

      // Save the flow to localStorage
      saveFlow(rf_inst);

      // Check how long the save took
      const duration = Date.now() - startTime;
      if (duration > 1500) {
        // If the operation took longer than 1.5 seconds, that's not good.
        // Although this function is called async inside setInterval,
        // calls to localStorage block the UI in JavaScript, freezing the screen.
        // We smart-disable autosaving here when we detect it's starting the freeze the UI:
        console.warn(
          "Autosaving disabled. The time required to save to localStorage exceeds 1 second. This can happen when there's a lot of data in your flow. Make sure to export frequently to save your work.",
        );
        clearInterval(interv);
        setAutosavingInterval(undefined);
      }
    }, 60000); // 60000 milliseconds = 1 minute
    setAutosavingInterval(interv);
  };

  // Run once upon ReactFlow initialization
  const onInit = (rf_inst: ReactFlowInstance) => {
    setRfInstance(rf_inst);

    if (IS_RUNNING_LOCALLY) {
      // If we're running locally, try to fetch API keys from Python os.environ variables in the locally running Flask backend:
      fetchEnvironAPIKeys()
        .then((api_keys) => {
          setAPIKeys(api_keys);
        })
        .catch((err) => {
          // Soft fail
          console.warn(
            "Warning: Could not fetch API key environment variables from Flask server. Error:",
            err.message,
          );
        });
    } else {
      // Check if there's a shared flow UID in the URL as a GET param
      // If so, we need to look it up in the database and attempt to load it:
      const shared_flow_uid = getSharedFlowURLParam();
      if (shared_flow_uid !== undefined) {
        try {
          // The format passed a basic smell test;
          // now let's query the server for a flow with that UID:
          fetch("/db/get_sharedflow.php", {
            method: "POST",
            body: shared_flow_uid,
          })
            .then((r) => r.text())
            .then((response) => {
              if (!response || response.startsWith("Error")) {
                // Error encountered during the query; alert the user
                // with the error message:
                throw new Error(response || "Unknown error");
              }

              // Attempt to parse the response as a compressed flow + import it:
              const cforge_json = JSON.parse(
                LZString.decompressFromUTF16(response),
              );
              importFlowFromJSON(cforge_json, rf_inst);
            })
            .catch(handleError);
        } catch (err) {
          // Soft fail
          setIsLoading(false);
          console.error(err);
        }

        // Since we tried to load from the shared flow ID, don't try to load from autosave
        return;
      }
    }

    // Attempt to load an autosaved flow, if one exists:
    if (autosavedFlowExists()) {
      const localStorageContent = localStorage.getItem("current_usecase");
      if (localStorageContent !== null) {
        const parsedData = JSON.parse(localStorageContent);
        // when user refresh the page and there is no file present, then we will reset the flow
        if (parsedData.file_name === "") {
          resetFlowToBlankCanvas();
        }
      } else {
        loadFlowFromAutosave(rf_inst);
      }
    } else {
      // Load an interesting default starting flow for new users
      importFlowFromJSON(EXAMPLEFLOW_1, rf_inst);

      // Open a welcome pop-up
      // openWelcomeModal();
    }

    // Turn off loading wheel
    setIsLoading(false);
  };

  const handleCreateUseCase = async (saveAndCommit: boolean) => {
    if (menuData && menuData.length === 0 && saveAndCommit) {
      return;
    }
    setLoading(true);
    try {
      if (saveAndCommit) {
        handleSaveFlow(saveAndCommit);
      }
      const localStorageData: any = localStorage.getItem("current_usecase");
      const aggragUserId = localStorage.getItem("aggrag-userId");
      const currUseCase = JSON.parse(localStorageData);
      const response = await fetch(`${API_URL}app/createusecase`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          folderName: saveAndCommit
            ? currUseCase.parent_folder
            : useCaseName.length > 0
              ? useCaseName
              : "usecase" + Date.now(),
          aggrag_user_id: aggragUserId,
          // prev_iteration: saveAndCommit ? currUseCase.iter_folder : "",
        }),
      });
      const res = await response.json();

      if (response.ok) {
        // this "if statement" is for fetching folder names when we create
        if (!isUseCaseCreated) {
          setIsUseCaseCreated(response.ok);
        } else {
          setIsUseCaseCreated(false);
        }
        setOpenCreateUseCase(false);
        setLoading(false);
        setOpenCreateUseCase(false);
        setUseCaseName("");
        setSaveDropdown("Save");
        const temp_p_folder = saveAndCommit
          ? currUseCase.parent_folder
          : res.usecase_folder;
        const temp_iter_folder = saveAndCommit
          ? res.iter_folder_name
          : "iteration 1";
        localStorage.setItem(
          "current_usecase",
          JSON.stringify({
            parent_folder: temp_p_folder !== undefined ? temp_p_folder : "",
            iter_folder: temp_iter_folder !== undefined ? temp_iter_folder : "",
            file_name: "",
            committed: false,
          }),
        );
        // here as we are creating new use case and iteration, so we can put false in committed value
        setActiveUseCase({
          usecase: temp_p_folder,
          iteration: temp_iter_folder,
          fileName: "",
          committed: false,
        });

        localStorage.setItem(
          "iteration-created",
          JSON.stringify({
            usecase: temp_p_folder,
            iteration: temp_iter_folder,
            fileName: "",
            committed: false,
            iterationCreated: true,
          }),
        );

        const queryString = `?p_folder=${encodeURIComponent(temp_p_folder)}&i_folder=${temp_iter_folder}&file_name=${encodeURIComponent("")}`;
        setIsCurrentFileLocked(false);
        // Update the URL
        window.history.pushState({}, "", queryString);
        showNotification("Created!", "Use case has been successfully created");
        // resetFlow();
        resetFlowToBlankCanvas();
        setIsChangesNotSaved(true);
      } else {
        setLoading(false);
        console.log("error in creating usecase");
        showNotification("Failed", "Error in creating usecase!", "red");
        setErrorMessage({ error: true, message: res.message });
      }
    } catch (error) {
      setLoading(false);
      console.log("error in creating usecase");
      showNotification("Failed", "Error in creating usecase!", "red");
      setErrorMessage({ error: true, message: "Error in creating usecase" });
    }
  };

  const handleUseCaseName = (value: string) => {
    if (value.length > 40) {
      setErrorMessage({
        error: true,
        message: "The use case name should not exceed 40 characters.",
      });
    } else {
      setUseCaseName(value);
      setErrorMessage({ error: false, message: "" });
    }
  };

  const handleEditUsecaseForCopy = (value: string) => {
    if (errorMessage.message) {
      setErrorMessage({ error: false, message: "" });
    }
    setEditUsecaseforCopy(value);
  };

  const handleSaveDropdown = () => {
    if (isCurrenFileLocked) {
      return;
    }
    if (saveDropdown === "Save") {
      handleSaveFlow(false);
    } else if (saveDropdown === "Save & Commit") {
      handleSaveAndCommit();
    }
  };

  const handleDeleteIteration = async (
    usecaseName: string,
    iterationName: string,
  ) => {
    setLoading(true);
    const response = await fetch(`${API_URL}app/deleteiteration`, {
      method: "DELETE",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        folderName: usecaseName,
        iterationFolder: iterationName,
      }),
    });
    if (response.ok) {
      if (!isUseCaseCreated) {
        setIsUseCaseCreated(response.ok);
      } else {
        setIsUseCaseCreated(false);
      }
      // this block of code is for showing deleted notification text only when user deletes.
      // (in some cases when undoing a iteration we are deleting iteration so in such cases we are skipping it)
      let parsedIterCreated = false;
      const iterCreated = localStorage.getItem("iteration-created");
      if (iterCreated !== null) {
        const parsedData = iterCreated && JSON.parse(iterCreated);
        parsedIterCreated = parsedData.iterationCreated;
      }
      if (!parsedIterCreated) {
        showNotification(
          "",
          `Iteration (${iterationName} of ${usecaseName && usecaseName.split("__")[0]}) has been successfully deleted!`,
          "red",
        );
      }
      // when a user creates an iteration and then clicks on another iteration without
      // saving previous iteration then
      // we are deleting unsaved iteration and redirecting to iteration which user has clicked.
      if (parsedIterCreated) {
        handleIterationFolderClick(
          modalOpen.usecase,
          modalOpen.iteration,
          modalOpen.subItems,
        );
        localStorage.setItem(
          "current_usecase",
          JSON.stringify({
            parent_folder: modalOpen.usecase,
            iter_folder: modalOpen.iteration,
            file_name: modalOpen.subItems && modalOpen.subItems[0]?.label,
            committed: false,
          }),
        );
        setActiveUseCase({
          fileName: modalOpen.subItems && modalOpen.subItems[0]?.label,
          iteration: modalOpen.iteration,
          usecase: modalOpen.usecase,
          committed: false,
        });
      } else {
        localStorage.setItem(
          "current_usecase",
          JSON.stringify({
            parent_folder: usecaseName,
            iter_folder: "",
            file_name: "",
            committed: false,
          }),
        );
        setActiveUseCase({
          fileName: "",
          iteration: "",
          usecase: usecaseName,
          committed: false,
        });
      }

      setLoading(false);
      setDeleteUsecaseOrIter({
        usecase: "",
        iteration: "",
        open: false,
      });

      localStorage.setItem(
        "iteration-created",
        JSON.stringify({
          usecase: "",
          iteration: "",
          fileName: "",
          committed: false,
          iterationCreated: false,
        }),
      );
      // we can keep committed value as false

      // Update the URL
      window.history.pushState({}, "", "/");
      setOpenMenu(false);
      resetFlowToBlankCanvas();
      setWarning({ warning: "", open: true });
      setIsCurrentFileLocked(false);
      setIsChangesNotSaved(false);
    }
  };

  const handleDelIterationForDefaultAndNormal = (subItem: any, item: any) => {
    // Check if item.label is defined
    if (!item.label) {
      return;
    }

    const newLabel = item.label.split("__");
    // If the label is 'default', exit the function
    if (newLabel.length > 1 && newLabel[1] === "default") {
      return;
    }

    // If the subItem has a committed subItem, exit the function
    if (subItem?.subItems?.[0]?.isCommitted) {
      return;
    }

    // Otherwise, set the delete use case or iteration
    setDeleteUsecaseOrIter({
      usecase: item.label,
      iteration: subItem.label,
      open: true,
    });
  };

  const handleIterationCopyForDefault = (subItem: any, item: any) => {
    if (item.label) {
      if (
        subItem &&
        subItem.subItems[0]?.isCommitted &&
        !(item.label.indexOf("__default") > -1)
      ) {
        handleCopyIteration(
          item.label,
          subItem.label,
          subItem.subItems[0].label,
        );
      }
    }
  };

  const handleCreateIteration = async (folderName: string) => {
    if (folderName.indexOf("__default") > -1) {
      return;
    }
    setOpenMenu(false);
    if (isChangesNotSaved) {
      setModalOpen({
        usecase: folderName,
        iteration: "",
        subItems: [],
        open: true,
        for: "create-iteration",
      });
      return;
    } else {
      setModalOpen({
        usecase: "",
        iteration: "",
        subItems: [],
        open: false,
        for: "",
      });
    }
    try {
      const response = await fetch(`${API_URL}app/createiteration`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          folderName: folderName,
        }),
      });

      const res: any = await response.json();
      if (response.ok) {
        if (!isUseCaseCreated) {
          setIsUseCaseCreated(response.ok);
        } else {
          setIsUseCaseCreated(false);
        }
        setOpenCreateUseCase(false);
        setLoading(false);
        setIsCurrentFileLocked(false);
        setIsChangesNotSaved(true);
        localStorage.setItem(
          "current_usecase",
          JSON.stringify({
            parent_folder: folderName,
            iter_folder: res.iter_folder_name,
            file_name: "",
            committed: false,
          }),
        );
        // here we can put committed value as false
        setActiveUseCase({
          usecase: folderName,
          iteration: res.iter_folder_name,
          fileName: "",
          committed: false,
        });
        const queryString = `?p_folder=${encodeURIComponent(folderName)}&i_folder=${res.iter_folder_name}&file_name=${encodeURIComponent("")}`;
        // Update the URL
        window.history.pushState({}, "", queryString);
        setSaveAndCommitBtnOpen(false);
        resetFlowToBlankCanvas();
        setOpenMenu(false);
        showNotification("Created", "Iteration has been created successfully");
        localStorage.setItem(
          "iteration-created",
          JSON.stringify({
            usecase: folderName,
            iteration: res.iter_folder_name,
            fileName: "",
            committed: false,
            iterationCreated: true,
          }),
        );
        // handleSaveFlow(false);
      }
    } catch (e) {
      console.log("error in creating iteration");
      showNotification("Failed", "Error in creating iteration", "red");
    }
  };

  const fetchFoldersAndContents = async () => {
    try {
      const aggragUserId = localStorage.getItem("aggrag-userId");
      const response = await fetch(
        `${API_URL}app/loadcforge?aggrag_user_id=${aggragUserId}`,
        {
          method: "GET",
        },
      );

      if (!response.ok) {
        throw new Error("Failed to fetch folders");
      }
      const data = await response.json();
      if (data) {
        setMenuData(data);
      }
    } catch (error) {
      console.error("Error fetching folders:", error);
    }
  };

  const handleCopyIteration = async (
    usecaseName: string,
    iterationName: string,
    fileName: string,
  ) => {
    if (isChangesNotSaved) {
      setModalOpen({
        usecase: usecaseName,
        iteration: iterationName,
        subItems: [
          {
            isCommitted: false,
            label: fileName,
          },
        ],
        open: true,
        for: "copy-iteration",
      });
      return;
    } else {
      setModalOpen({
        usecase: "",
        iteration: "",
        subItems: [],
        open: false,
        for: "",
      });
    }

    setOpenMenu(false);
    const response = await fetch(`${API_URL}app/copyiteration`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        folderName: usecaseName,
        iterationName: iterationName,
        fileName: fileName,
      }),
    });
    const res: any = await response.json();
    if (response.ok) {
      // this "if statement" is for fetching folder names when we create
      if (!isUseCaseCreated) {
        setIsUseCaseCreated(response.ok);
      } else {
        setIsUseCaseCreated(false);
      }
      setOpenCreateUseCase(false);
      setLoading(false);
      setIsCurrentFileLocked(false);
      // setIsChangesNotSaved(true);
      localStorage.setItem(
        "current_usecase",
        JSON.stringify({
          parent_folder: usecaseName,
          iter_folder: res.iter_folder_name,
          file_name: res.fileName ? res.fileName : "",
          committed: false,
        }),
      );
      // here also we can put committed value as false
      setActiveUseCase({
        usecase: usecaseName,
        iteration: res.iter_folder_name,
        fileName: res.file_name,
        committed: false,
      });
      const queryString = `?p_folder=${encodeURIComponent(usecaseName)}&i_folder=${res.iter_folder_name}&file_name=${encodeURIComponent(res.file_name)}`;
      // Update the URL
      window.history.pushState({}, "", queryString);
      setSaveAndCommitBtnOpen(false);
      setOpenMenu(false);
      // here loading the iteration folder and file
      handleIterationFolderClick(usecaseName, res.iter_folder_name, [
        {
          isCommitted: false,
          label: res.file_name,
        },
      ]);
      showNotification("Copied", `Iteration has been copied successfully`);
    }
  };

  const handleDeleteUsecase = async (usecasename: string) => {
    setLoading(true);
    const response = await fetch(`${API_URL}app/deleteusecase`, {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ usecasename: usecasename }),
    });
    if (response.ok) {
      setLoading(false);

      if (!isUseCaseCreated) {
        setIsUseCaseCreated(response.ok);
      } else {
        setIsUseCaseCreated(false);
      }
      localStorage.setItem(
        "current_usecase",
        JSON.stringify({
          parent_folder: "",
          iter_folder: "",
          file_name: "",
          committed: false,
        }),
      );
      // committed value as false
      setActiveUseCase({
        usecase: "",
        iteration: "",
        fileName: "",
        committed: false,
      });
      setLoading(false);
      // Update the URL
      window.history.pushState({}, "", "/");
      setDeleteUsecaseOrIter({ usecase: "", iteration: "", open: false });
      showNotification(
        "",
        `Use case (${usecasename && usecasename.split("__")[0]}) has been successfully deleted!`,
      );
      resetFlowToBlankCanvas();
      setOpenMenu(false);
      setWarning({ warning: "", open: true });
      setIsCurrentFileLocked(true);
      setIsChangesNotSaved(false);
    }
  };

  const handleCopyUsecaseModal = (item: any) => {
    setCopyModalOpen({
      for: "copy-usecase",
      open: true,
      usecase: item.label,
    });
  };

  const handleCopyUsecase = async (usecaseName: any) => {
    setLoading(true);
    setEditUsecaseforCopy("");
    const aggragUserId = localStorage.getItem("aggrag-userId");
    const response = await fetch(`${API_URL}app/copyusecase`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        sourceUsecase: usecaseName,
        targetUsecase: editUsecaseforCopy + "__" + aggragUserId,
        aggrag_user_id: aggragUserId,
      }),
    });
    const usecaseResponse = await response.json();
    if (response.ok) {
      if (!isUseCaseCreated) {
        setIsUseCaseCreated(response.ok);
      } else {
        setIsUseCaseCreated(false);
      }
      showNotification("Copied", "Usecase has been successfully copied");
      setCopyModalOpen({
        for: "",
        open: false,
        usecase: "",
      });
      const res_file_name =
        usecaseResponse.iterations_info.length > 0 &&
        usecaseResponse.iterations_info[0].files[0]
          ? usecaseResponse.iterations_info[0].files[0]
          : "";
      const res_iteration_name =
        usecaseResponse.iterations_info.length > 0
          ? usecaseResponse.iterations_info[0].iteration_name
          : "";
      setActiveUseCase({
        usecase: usecaseResponse.target_usecase_folder_name,
        iteration: res_iteration_name,
        fileName: res_file_name,
        committed: false,
      });
      localStorage.setItem(
        "current_usecase",
        JSON.stringify({
          parent_folder: usecaseResponse.target_usecase_folder_name,
          iter_folder: res_iteration_name,
          file_name: res_file_name,
          committed: false,
        }),
      );
      handleIterationFolderClick(
        usecaseResponse.target_usecase_folder_name,
        res_iteration_name,
        [
          {
            isCommitted: false,
            label: res_file_name,
          },
        ],
      );
      setEditUsecaseforCopy("");
    } else {
      setErrorMessage({ error: true, message: usecaseResponse.message });
    }
    setLoading(false);
  };

  const getLabelDefault = (label: string) => {
    if (label) {
      // const newLabel = label.split("__")[1].toLowerCase();
      const newLabel = label.split("__");

      if (newLabel.length > 1 && newLabel[1].toLowerCase() === "default") {
        return (
          <span
            style={{
              fontSize: 8,
              color: "green",
              border: "1px green solid",
              textAlign: "center",
              padding: "0px 6px 2px 6px",
              borderRadius: "8px",
              position: "absolute",
              marginLeft: "5px",
              marginTop: "3px",
            }}
          >
            example
          </span>
        );
      }
    }
  };

  const removeBinForDefaults = (label: string) => {
    if (label) {
      const newLabel = label.split("__");

      if (newLabel.length > 1 && newLabel[1].toLowerCase() === "default") {
        return (
          <div style={{ cursor: "not-allowed" }}>
            <BinIcon color="#7b8990" />
          </div>
        );
      } else {
        return <BinIcon />;
      }
    }
  };

  const getLabelName = (label: string) => {
    if (label) {
      const newLabel = label.split("__");
      if (newLabel) {
        const items = newLabel.slice(0, -1);
        const filteredItems = items.filter(
          (item) => !item.includes("__default"),
        );
        const finalNameString = filteredItems.join("__");
        return finalNameString;
      } else {
        return "";
      }
    } else {
      return "";
    }
  };

  const handleDisableSave = () => {
    if (activeUseCase && activeUseCase.usecase.split("__")[1] === "default") {
      return true;
    } else {
      return false;
    }
  };

  const renderBinIcon = (item: any, subItem: any) => {
    if (subItem.subItems && subItem.subItems[0]?.isCommitted) {
      return false;
    } else if (
      item.label &&
      item.label.split("__").length > 1 &&
      item.label.split("__")[1].toLowerCase() === "default"
    ) {
      return false;
    }
    return true;
  };

  const renderCopyIcon = (item: any, subItem: any) => {
    if (
      item.label &&
      item.label.split("__").length > 1 &&
      item.label.split("__")[1].toLowerCase() === "default"
    ) {
      return false;
    } else if (subItem.subItems && subItem.subItems[0]?.isCommitted) {
      return true;
    } else {
      return false;
    }
  };

  // this code is for routing to respective methods when user confirms in the save changes modal
  useEffect(() => {
    if (confirmed) {
      if (modalOpen.for === "create-iteration") {
        const data = localStorage.getItem("iteration-created");
        const parsedData = data && JSON.parse(data);
        if (parsedData && parsedData.iterationCreated) {
          handleDeleteIteration(parsedData.usecase, parsedData.iteration);
        }
        handleCreateIteration(modalOpen.usecase);
      } else if (modalOpen.for === "copy-iteration") {
        handleCopyIteration(
          modalOpen.usecase,
          modalOpen.iteration,
          modalOpen.subItems[0]?.label,
        );
      } else if (modalOpen.for === "click-iteration") {
        const data = localStorage.getItem("iteration-created");
        const parsedData = data && JSON.parse(data);
        if (parsedData && parsedData.iterationCreated) {
          handleDeleteIteration(parsedData.usecase, parsedData.iteration);
          console.log("triggered");
        } else {
          handleIterationFolderClick(
            modalOpen.usecase,
            modalOpen.iteration,
            modalOpen.subItems,
          );
        }
      } else if (copyModalOpen.for === "copy-usecase") {
        handleCopyUsecase(copyModalOpen.usecase);
      }

      setConfirmed(false);
    }
  }, [confirmed]);

  useEffect(() => {
    const localStorageContent = localStorage.getItem("current_usecase");
    const userId = localStorage.getItem("aggrag-userId");

    if (userId === null) {
      const currentTimeStamp = new Date().getTime();
      localStorage.setItem("aggrag-userId", currentTimeStamp.toString());
    }

    if (localStorageContent != null) {
      const parsedData = JSON.parse(localStorageContent);

      if (
        parsedData &&
        parsedData.file_name &&
        parsedData.file_name.length > 0
      ) {
        try {
          loadExistingFile(
            parsedData.parent_folder,
            parsedData.iter_folder,
            parsedData.file_name,
            parsedData.committed,
          );
        } catch (e) {
          console.log("error in retrieving file data");
        }
      } else {
        resetFlowToBlankCanvas();
      }

      setActiveUseCase({
        usecase: parsedData.parent_folder,
        iteration: parsedData.iter_folder,
        fileName: parsedData.file_name,
        committed: parsedData.committed,
      });

      if (parsedData.committed) {
        setIsCurrentFileLocked(true);
      }
    }
    if (window) {
      window.onerror = function (message, source, lineno, colno, error) {
        // Log the error details to a remote server
        const errorData = {
          message,
          source,
          lineno,
          colno,
          error: error ? error.stack : null,
        };
        console.error("Global error logging: ", errorData);
        // Send errorData to a logging service (e.g., via an HTTP request)
        if (
          typeof errorData.message === "string" &&
          errorData.message.startsWith("ResizeObserver loop")
        ) {
          console.warn(
            "Ignored: ResizeObserver loop error: ",
            errorData.message,
          );
          return false;
        } else {
          showNotification(
            "Failed",
            errorData && errorData.message.toString(),
            "red",
          );
          // Return true to prevent the default browser error handling
          return true;
        }
      };

      // Handle unhandled promise rejections
      window.addEventListener("unhandledrejection", function (event) {
        const errorData = {
          message: event.reason.message,
          stack: event.reason.stack,
        };
        console.error("Unhandled promise rejection: ", errorData);
        showNotification("Failed", errorData.message, "red");
        // Prevent the default handling (e.g., logging to the console)
        event.preventDefault();
      });
    }
  }, []);

  useEffect(() => {
    // Cleanup the autosaving interval upon component unmount:
    return () => {
      clearInterval(autosavingInterval); // Clear the interval when the component is unmounted
    };
  }, []);

  useEffect(() => {
    window.addEventListener("error", (e) => {
      if (e.message.startsWith("ResizeObserver loop")) {
        const resizeObserverErrDiv = document.getElementById(
          "webpack-dev-server-client-overlay-div",
        );
        const resizeObserverErr = document.getElementById(
          "webpack-dev-server-client-overlay",
        );
        if (resizeObserverErr) {
          resizeObserverErr.setAttribute("style", "display: none");
        }
        if (resizeObserverErrDiv) {
          resizeObserverErrDiv.setAttribute("style", "display: none");
        }
      }
    });
  }, []);

  useEffect(() => {
    fetchFoldersAndContents();
  }, [isUseCaseCreated]);

  if (!IS_ACCEPTED_BROWSER) {
    return (
      <Box maw={600} mx="auto" mt="40px">
        <Text m="xl" size={"11pt"}>
          {"We're sorry, but it seems like "}
          {isMobile
            ? "you are viewing Aggrag on a mobile device"
            : "your current browser isn't supported by the current version of Aggrag"}{" "}
          😔. We want to provide you with the best experience possible, so we
          recommend {isMobile ? "viewing Aggrag on a desktop browser" : ""}{" "}
          using one of our supported browsers listed below:
        </Text>
        <List m="xl" size={"11pt"}>
          <List.Item>Google Chrome</List.Item>
          <List.Item>Mozilla Firefox</List.Item>
          <List.Item>Microsoft Edge (Chromium)</List.Item>
          <List.Item>Brave</List.Item>
        </List>

        <Text m="xl" size={"11pt"}>
          These browsers offer enhanced compatibility with Aggrag&apos;s
          features. Don&apos;t worry, though! We&apos;re working to expand our
          browser support to ensure everyone can enjoy our platform. 😊
        </Text>
        <Text m="xl" size={"11pt"}>
          If you have any questions or need assistance, please don&apos;t
          hesitate to reach out on our{" "}
          <a href="https://github.com/genai-apps/aggrag/issues">GitHub</a> by{" "}
          <a href="https://github.com/genai-apps/aggrag/issues">
            opening an Issue.
          </a>
          &nbsp; (If you&apos;re a web developer, consider forking our
          repository and making a{" "}
          <a href="https://github.com/genai-apps/aggrag/pulls">Pull Request</a>{" "}
          to support your particular browser.)
        </Text>
      </Box>
    );
  } else
    return (
      <div>
        <GlobalSettingsModal ref={settingsModal} />
        <LoadingOverlay visible={isLoading} overlayBlur={1} />
        <ExampleFlowsModal
          ref={examplesModal}
          handleOnSelect={onSelectExampleFlow}
        />
        <AreYouSureModal
          ref={confirmationModal}
          title={confirmationDialogProps.title}
          message={confirmationDialogProps.message}
          onConfirm={confirmationDialogProps.onConfirm}
        />
        <Modal
          opened={copyModalOpen.open}
          onClose={() => {
            setCopyModalOpen({
              for: "",
              open: false,
              usecase: "",
            });
          }}
          title={<div style={{ fontWeight: "500" }}>Copy Use case</div>}
          styles={{
            header: { backgroundColor: "#228be6", color: "white" },
            root: { position: "relative", left: "-5%" },
            close: {
              color: "#fff",
              "&:hover": {
                color: "black",
              },
            },
          }}
        >
          <Box maw={400} mx="auto" mt="md" mb="md">
            <Text>Use case name:</Text>
            <Input
              value={editUsecaseforCopy}
              onChange={(e: any) => handleEditUsecaseForCopy(e.target.value)}
              style={{ marginTop: "6px", marginBottom: "6px" }}
              title="Use case name"
            />
            <div
              style={{ color: "red", marginBottom: "10px", fontSize: "12px" }}
            >
              {errorMessage.message}
            </div>
            <Text>Description: </Text>
            <Textarea
              placeholder="Use Case Description (optional)"
              style={{ marginTop: "10px" }}
            />
          </Box>
          <Flex
            mih={50}
            gap="md"
            justify="space-evenly"
            align="center"
            direction="row"
            wrap="wrap"
          >
            <Button
              variant="light"
              color="orange"
              type="submit"
              w="40%"
              onClick={() =>
                setCopyModalOpen({
                  for: "",
                  open: false,
                  usecase: "",
                })
              }
            >
              Cancel
            </Button>
            <Button
              variant="filled"
              color="blue"
              type="submit"
              w="40%"
              onClick={() => {
                setConfirmed(true);
              }}
              disabled={!(editUsecaseforCopy.length > 0)}
              loading={loading}
            >
              Confirm
            </Button>
          </Flex>
        </Modal>

        <Modal
          transitionProps={{ transition: "pop" }}
          title="Changes are not saved"
          opened={modalOpen.open}
          onClose={() =>
            setModalOpen({
              usecase: "",
              iteration: "",
              subItems: [],
              open: false,
              for: "",
            })
          }
          styles={{
            root: { position: "relative", left: "-5%" },
          }}
        >
          Are you sure you want to proceed? Any unsaved changes will be lost.
          <div style={{ display: "flex", gap: "12px", justifyContent: "end" }}>
            <Button
              variant="outline"
              disabled={false}
              onClick={() =>
                setModalOpen({
                  usecase: "",
                  iteration: "",
                  subItems: [],
                  open: false,
                  for: "",
                })
              }
              loading={loading}
            >
              Cancel
            </Button>
            <Button
              disabled={false}
              onClick={() => {
                setIsChangesNotSaved(false);
                setConfirmed(true);
              }}
              loading={loading}
            >
              Confirm
            </Button>
          </div>
        </Modal>

        <Modal
          transitionProps={{ transition: "pop" }}
          title="Create a use case"
          opened={openCreateUseCase}
          onClose={() => setOpenCreateUseCase(false)}
          styles={{
            root: { position: "relative", left: "-5%" },
          }}
        >
          <TextInput
            value={useCaseName}
            onChange={(event) => handleUseCaseName(event.target.value)}
            className="usecase-input"
            placeholder="Name"
            label={"Use case name:"}
            radius={"md"}
          />
          <Textarea
            value={description}
            onChange={(event) => setDescription(event.target.value)}
            placeholder="Use Case Description (optional)"
            style={{ marginTop: "10px" }}
            label={"Description:"}
          />
          <div style={{ color: "red", marginBottom: "10px", fontSize: "12px" }}>
            {errorMessage.message}
          </div>

          <div style={{ display: "flex", justifyContent: "center" }}>
            <Button
              disabled={!(useCaseName.length > 0)}
              onClick={() => handleCreateUseCase(false)}
              loading={loading}
            >
              Confirm
            </Button>
          </div>
        </Modal>

        <Modal
          transitionProps={{ transition: "pop" }}
          title={
            <b>
              Delete{" "}
              {deleteusecaseOrIter.usecase.length > 0 &&
              deleteusecaseOrIter.iteration.length > 0
                ? "Iteration"
                : "Use case"}
            </b>
          }
          opened={deleteusecaseOrIter.open}
          onClose={() =>
            setDeleteUsecaseOrIter({
              usecase: "",
              iteration: "",
              open: false,
            })
          }
          styles={{
            root: { position: "relative", left: "-5%" },
          }}
        >
          Are you sure you want to delete this{" "}
          <b>
            {deleteusecaseOrIter.usecase.length > 0 &&
            deleteusecaseOrIter.iteration.length > 0
              ? deleteusecaseOrIter.iteration
              : deleteusecaseOrIter.usecase.split("__")[0]}
          </b>
          <div
            style={{
              display: "flex",
              justifyContent: "end",
              marginTop: "16px",
              gap: "12px",
            }}
          >
            <Button
              variant="outline"
              onClick={() =>
                setDeleteUsecaseOrIter({
                  usecase: "",
                  iteration: "",
                  open: false,
                })
              }
            >
              Cancel
            </Button>
            <Button
              onClick={() => {
                deleteusecaseOrIter.usecase.length > 0 &&
                deleteusecaseOrIter.iteration.length > 0
                  ? handleDeleteIteration(
                      deleteusecaseOrIter.usecase,
                      deleteusecaseOrIter.iteration,
                    )
                  : handleDeleteUsecase(deleteusecaseOrIter.usecase);
              }}
              loading={loading}
            >
              Confirm
            </Button>
          </div>
        </Modal>

        {/* <Modal title={'Welcome to Aggrag'} size='400px' opened={welcomeModalOpened} onClose={closeWelcomeModal} yOffset={'6vh'} styles={{header: {backgroundColor: '#FFD700'}, root: {position: 'relative', left: '-80px'}}}>
          <Box m='lg' mt='xl'>
            <Text>To get started, click the Settings icon in the top-right corner.</Text>
          </Box>
        </Modal> */}

        <div
          id="cf-root-container"
          style={{ display: "flex", height: "100vh" }}
          onPointerDown={hideContextMenu}
          onClick={() => {
            setOpenAddNode(false);
            setOpenMenu(false);
            setSaveAndCommitBtnOpen(false);
          }}
        >
          <div
            style={{ height: "100%", backgroundColor: "#eee", flexGrow: "1" }}
          >
            <ReactFlow
              minZoom={0.7}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onConnect={onConnect}
              nodes={nodes}
              edges={edges}
              // @ts-expect-error Node types won't perfectly fit unless we explicitly extend from RF's types; ignoring this for now.
              nodeTypes={nodeTypes}
              // @ts-expect-error Edge types won't perfectly fit unless we explicitly extend from RF's types; ignoring this for now.
              edgeTypes={edgeTypes}
              zoomOnPinch={false}
              zoomOnScroll={false}
              panOnScroll={true}
              disableKeyboardA11y={true}
              deleteKeyCode={[]}
              // connectionLineComponent={AnimatedConnectionLine}
              // connectionLineStyle={connectionLineStyle}
              snapToGrid={true}
              snapGrid={snapGrid}
              onInit={onInit}
              onError={() => {
                // Suppress ReactFlow warnings spamming the console.
                // console.log(err);
              }}
              edgesUpdatable={!isCurrenFileLocked}
              edgesFocusable={!isCurrenFileLocked}
              nodesDraggable={!isCurrenFileLocked}
              // nodesConnectable={!isCurrenFileLocked}
              nodesFocusable={!isCurrenFileLocked}
              // draggable={!isCurrenFileLocked}
              elementsSelectable={!isCurrenFileLocked}
            >
              <Background color="#999" gap={16} />
              <Controls showZoom={true} />
              {/* <MiniMap zoomable pannable /> */}
            </ReactFlow>
          </div>
        </div>
        <Tooltip
          multiline
          width={300}
          withinPortal
          withArrow
          label={
            isCurrenFileLocked &&
            "This is a commited iteration, hence can not be edited. Create a new/duplicate to continue."
          }
          style={{
            backgroundColor: isCurrenFileLocked
              ? (activeUseCase && activeUseCase.usecase === "") ||
                (activeUseCase && activeUseCase.iteration === "")
                ? "transparent"
                : ""
              : "transparent",
            color:
              (activeUseCase && activeUseCase.usecase === "") ||
              (activeUseCase && activeUseCase.iteration === "")
                ? "transparent"
                : "",
          }}
        >
          <div className="top-bar">
            <div id="custom-controls">
              <Menu
                transitionProps={{ transition: "pop-top-left" }}
                position="top-start"
                closeOnClickOutside={true}
                closeOnEscape
                trigger="click"
                width={270}
                opened={openMenu}
              >
                <Menu.Target>
                  <Button
                    size="sm"
                    compact
                    mr="sm"
                    onClick={() => {
                      setSaveAndCommitBtnOpen(false);
                      setOpenMenu(!openMenu);
                      setOpenAddNode(false);
                    }}
                    variant="gradient"
                  >
                    Use Cases +
                  </Button>
                </Menu.Target>

                <Menu.Dropdown>
                  <div style={{ overflowY: "auto", maxHeight: 600 }}>
                    <Menu.Item
                      onClick={() => {
                        setUseCaseName("");
                        setOpenMenu(false);
                        setDescription("");
                        setErrorMessage({ error: false, message: "" });
                        setOpenCreateUseCase(true);
                      }}
                      style={{
                        cursor: "pointer",
                        marginTop: "2px",
                        color: "#726C72",
                      }}
                    >
                      Create a new use case
                    </Menu.Item>

                    {menuData &&
                      menuData.map((item: any, index: any) => {
                        if (item.subItems) {
                          return (
                            <Menu
                              key={index}
                              transitionProps={{ transition: "pop-top-left" }}
                              trigger="hover"
                              width={300}
                              shadow="md"
                              position="right-start"
                            >
                              <Menu.Target>
                                <Menu.Item
                                  className="menu-hover-item"
                                  rightSection={
                                    <div
                                      style={{
                                        display: "flex",
                                        alignItems: "center",
                                        justifyContent: "center",
                                        gap: "10px",
                                      }}
                                    >
                                      <Tooltip label="Delete">
                                        <div
                                          onClick={() => {
                                            setOpenMenu(false);
                                            item.label &&
                                              item.label.split("__").length >
                                                1 &&
                                              item.label
                                                .split("__")[1]
                                                .toLowerCase() !== "default" &&
                                              setDeleteUsecaseOrIter({
                                                usecase: item.label,
                                                iteration: "",
                                                open: true,
                                              });
                                          }}
                                          style={{
                                            display: "flex",
                                            alignSelf: "center",
                                            marginTop: "3px",
                                          }}
                                        >
                                          {removeBinForDefaults(item.label)}
                                        </div>
                                      </Tooltip>
                                      <div
                                        style={{ marginTop: "4px" }}
                                        onClick={() =>
                                          handleCopyUsecaseModal(item)
                                        }
                                      >
                                        <CopyIcon />
                                      </div>
                                    </div>
                                  }
                                  style={{
                                    backgroundColor:
                                      activeUseCase.usecase === item.label ||
                                      hoveredItem === item.label
                                        ? "#ececec"
                                        : "",
                                    marginTop:
                                      activeUseCase.usecase === item.label ||
                                      hoveredItem === item.label
                                        ? "2px"
                                        : "",
                                    marginBottom:
                                      activeUseCase.usecase === item.label ||
                                      hoveredItem === item.label
                                        ? "2px"
                                        : "",
                                  }}
                                  onMouseEnter={() =>
                                    setHoveredItem(item.label)
                                  }
                                  onMouseLeave={() => setHoveredItem(null)}
                                  // onClick={() => handleUseCaseFolderClick(item.label)}
                                  styles={{
                                    root: {
                                      "&:hover": { backgroundColor: "#000" },
                                      "&:active": {
                                        backgroundColor: "#e0e0e0",
                                      },
                                    },
                                  }}
                                >
                                  {/* here we are separating label because from backend we are getting name and timestamp */}
                                  <span>
                                    {/* {item.label && item.label.split("__")[0]} */}
                                    {getLabelName(item.label)}
                                  </span>{" "}
                                  {item.label && getLabelDefault(item.label)}
                                </Menu.Item>
                              </Menu.Target>

                              <Menu.Dropdown style={{ width: "400px" }}>
                                <Menu.Item
                                  onClick={() => {
                                    setSaveAndCommitBtnOpen(false);
                                    // handleCreateNewIteration(item.label);
                                    handleCreateIteration(item.label);
                                  }}
                                >
                                  Create a new iteration
                                </Menu.Item>
                                {item.subItems.map(
                                  (subItem: any, subIndex: any) => {
                                    if (subItem.subItems) {
                                      return (
                                        <Menu
                                          key={subIndex}
                                          transitionProps={{
                                            transition: "pop-top-left",
                                          }}
                                          trigger="hover"
                                          width={200}
                                          shadow="md"
                                          position="right-start"
                                        >
                                          <Menu.Target>
                                            <Menu.Item
                                              onMouseEnter={() =>
                                                setHoveredItem(item.label)
                                              }
                                              onMouseLeave={() =>
                                                setHoveredItem(null)
                                              }
                                              className="menu-hover-item"
                                              rightSection={
                                                <div
                                                  style={{
                                                    display: "flex",
                                                    alignItems: "center",
                                                    gap: "12px",
                                                  }}
                                                >
                                                  <div
                                                    style={{
                                                      marginTop: "8px",
                                                    }}
                                                  >
                                                    {subItem.subItems &&
                                                      subItem.subItems[0]
                                                        ?.isCommitted && (
                                                        <LockIcon />
                                                      )}
                                                  </div>
                                                  <div
                                                    style={{
                                                      marginTop: "8px",
                                                    }}
                                                    onClick={(e) => {
                                                      e.stopPropagation();
                                                      setOpenMenu(false);
                                                      handleDelIterationForDefaultAndNormal(
                                                        subItem,
                                                        item,
                                                      );
                                                    }}
                                                  >
                                                    {renderBinIcon(
                                                      item,
                                                      subItem,
                                                    ) ? (
                                                      <BinIcon />
                                                    ) : (
                                                      <div
                                                        style={{
                                                          cursor: "not-allowed",
                                                        }}
                                                      >
                                                        <BinIcon
                                                          color={"#7b8990"}
                                                        />
                                                      </div>
                                                    )}
                                                  </div>
                                                  <div
                                                    style={{
                                                      display: "flex",
                                                      alignItems: "center",
                                                      gap: "12px",
                                                      marginTop: "4px",
                                                    }}
                                                  >
                                                    <div
                                                      style={{
                                                        marginTop: "2px",
                                                      }}
                                                      onClick={(e) => {
                                                        e.stopPropagation();
                                                        handleIterationCopyForDefault(
                                                          subItem,
                                                          item,
                                                        );
                                                      }}
                                                    >
                                                      {renderCopyIcon(
                                                        item,
                                                        subItem,
                                                      ) ? (
                                                        <CopyIcon />
                                                      ) : (
                                                        <div
                                                          style={{
                                                            cursor:
                                                              "not-allowed",
                                                          }}
                                                        >
                                                          <CopyIcon
                                                            color={"#7b8990"}
                                                          />
                                                        </div>
                                                      )}
                                                    </div>
                                                  </div>
                                                </div>
                                              }
                                              onClick={(e) => {
                                                e.stopPropagation();
                                                handleIterationFolderClick(
                                                  item.label,
                                                  subItem.label,
                                                  subItem.subItems,
                                                );
                                              }}
                                              style={{
                                                height: "32px",
                                                backgroundColor:
                                                  activeUseCase.usecase ===
                                                    item.label &&
                                                  activeUseCase.iteration ===
                                                    subItem.label
                                                    ? "#ececec"
                                                    : "",
                                              }}
                                            >
                                              <span className="overflow-iteration-name">
                                                {subItem.label}
                                              </span>
                                            </Menu.Item>
                                          </Menu.Target>
                                        </Menu>
                                      );
                                    }
                                    return (
                                      <Menu.Item key={subIndex}>
                                        {subItem.label}
                                      </Menu.Item>
                                    );
                                  },
                                )}
                              </Menu.Dropdown>
                            </Menu>
                          );
                        }

                        return <Menu.Item key={index}>{item.label}</Menu.Item>;
                      })}
                  </div>
                </Menu.Dropdown>
              </Menu>

              <Menu
                transitionProps={{ transition: "pop-top-left" }}
                position="top-start"
                width={220}
                closeOnClickOutside={true}
                closeOnEscape
                styles={{ item: { maxHeight: "28px" } }}
                disabled={isCurrenFileLocked}
                trigger="click"
                opened={openAddNode}
              >
                <Menu.Target>
                  <Button
                    size="sm"
                    variant="gradient"
                    compact
                    mr="sm"
                    disabled={
                      (activeUseCase && activeUseCase.usecase === "") ||
                      isCurrenFileLocked
                    }
                    onClick={() => {
                      setOpenAddNode(!openAddNode);
                      setOpenMenu(false);
                      setSaveAndCommitBtnOpen(false);
                    }} // to close use cases menu
                  >
                    Add Node +
                  </Button>
                </Menu.Target>
                <Menu.Dropdown>
                  <Menu.Label>Knowledge Base</Menu.Label>
                  <MenuTooltip label="Input file to prompt or chat nodes. You can also declare variables in brackets {} to chain FileFields together.">
                    <Menu.Item
                      onClick={addUploadFileFieldsNode}
                      icon={<IconForms size="16px" />}
                    >
                      {" "}
                      Upload File Fields Node{" "}
                    </Menu.Item>
                  </MenuTooltip>
                  <Menu.Divider />
                  <Menu.Label>Input Data</Menu.Label>
                  <MenuTooltip label="Specify input text to prompt or chat nodes. You can also declare variables in brackets {} to chain TextFields together.">
                    <Menu.Item
                      onClick={addTextFieldsNode}
                      icon={<IconTextPlus size="16px" />}
                    >
                      {" "}
                      TextFields Node{" "}
                    </Menu.Item>
                  </MenuTooltip>
                  <MenuTooltip label="Specify inputs as a comma-separated list of items. Good for specifying lots of short text values. An alternative to TextFields node.">
                    <Menu.Item
                      onClick={addItemsNode}
                      icon={<IconForms size="16px" />}
                    >
                      {" "}
                      Items Node{" "}
                    </Menu.Item>
                  </MenuTooltip>
                  <MenuTooltip label="Import or create a spreadhseet of data to use as input to prompt or chat nodes. Import accepts xlsx, csv, and jsonl.">
                    <Menu.Item onClick={addTabularDataNode} icon={"🗂️"}>
                      {" "}
                      Tabular Data Node{" "}
                    </Menu.Item>
                  </MenuTooltip>
                  <Menu.Divider />
                  <Menu.Label>Prompters</Menu.Label>
                  <MenuTooltip label="Prompt one or multiple LLMs. Specify prompt variables in brackets {}.">
                    <Menu.Item onClick={addPromptNode} icon={"💬"}>
                      {" "}
                      Prompt Node{" "}
                    </Menu.Item>
                  </MenuTooltip>
                  <MenuTooltip label="Start or continue a conversation with chat models. Attach Prompt Node output as past context to continue chatting past the first turn.">
                    <Menu.Item onClick={addChatTurnNode} icon={"🗣"}>
                      {" "}
                      Chat Turn Node{" "}
                    </Menu.Item>
                  </MenuTooltip>
                  <Menu.Divider />
                  <Menu.Label>Evaluators</Menu.Label>
                  <MenuTooltip label="Evaluate responses with a simple check (no coding required).">
                    <Menu.Item
                      onClick={addSimpleEvalNode}
                      icon={<IconRuler2 size="16px" />}
                    >
                      {" "}
                      Simple Evaluator{" "}
                    </Menu.Item>
                  </MenuTooltip>
                  <MenuTooltip label="Evaluate responses by writing JavaScript code.">
                    <Menu.Item
                      onClick={() => addEvalNode("javascript")}
                      icon={<IconTerminal size="16px" />}
                    >
                      {" "}
                      JavaScript Evaluator{" "}
                    </Menu.Item>
                  </MenuTooltip>
                  <MenuTooltip label="Evaluate responses by writing Python code.">
                    <Menu.Item
                      onClick={() => addEvalNode("python")}
                      icon={<IconTerminal size="16px" />}
                    >
                      {" "}
                      Python Evaluator{" "}
                    </Menu.Item>
                  </MenuTooltip>
                  <MenuTooltip label="Evaluate responses with an LLM like GPT-4.">
                    <Menu.Item
                      onClick={addLLMEvalNode}
                      icon={<IconRobot size="16px" />}
                    >
                      {" "}
                      LLM Scorer{" "}
                    </Menu.Item>
                  </MenuTooltip>
                  <MenuTooltip label="Evaluate responses across multiple criteria (multiple code and/or LLM evaluators).">
                    <Menu.Item
                      onClick={addMultiEvalNode}
                      icon={<IconAbacus size="16px" />}
                    >
                      {" "}
                      Multi-Evaluator{" "}
                    </Menu.Item>
                  </MenuTooltip>
                  <Menu.Divider />
                  <Menu.Label>Visualizers</Menu.Label>
                  <MenuTooltip label="Plot evaluation results. (Attach an evaluator or scorer node as input.)">
                    <Menu.Item onClick={addVisNode} icon={"📊"}>
                      {" "}
                      Vis Node{" "}
                    </Menu.Item>
                  </MenuTooltip>
                  <MenuTooltip label="Used to inspect responses from prompter or evaluation nodes, without opening up the pop-up view.">
                    <Menu.Item onClick={addInspectNode} icon={"🔍"}>
                      {" "}
                      Inspect Node{" "}
                    </Menu.Item>
                  </MenuTooltip>
                  <Menu.Divider />
                  <Menu.Label>Processors</Menu.Label>
                  <MenuTooltip label="Transform responses by mapping a JavaScript function over them.">
                    <Menu.Item
                      onClick={() => addProcessorNode("javascript")}
                      icon={<IconTerminal size="14pt" />}
                    >
                      {" "}
                      JavaScript Processor{" "}
                    </Menu.Item>
                  </MenuTooltip>
                  {IS_RUNNING_LOCALLY ? (
                    <MenuTooltip label="Transform responses by mapping a Python function over them.">
                      <Menu.Item
                        onClick={() => addProcessorNode("python")}
                        icon={<IconTerminal size="14pt" />}
                      >
                        {" "}
                        Python Processor{" "}
                      </Menu.Item>
                    </MenuTooltip>
                  ) : (
                    <></>
                  )}
                  <MenuTooltip label="Concatenate responses or input data together before passing into later nodes, within or across variables and LLMs.">
                    <Menu.Item
                      onClick={addJoinNode}
                      icon={<IconArrowMerge size="14pt" />}
                    >
                      {" "}
                      Join Node{" "}
                    </Menu.Item>
                  </MenuTooltip>
                  <MenuTooltip label="Split responses or input data by some format. For instance, you can split a markdown list into separate items.">
                    <Menu.Item
                      onClick={addSplitNode}
                      icon={<IconArrowsSplit size="14pt" />}
                    >
                      {" "}
                      Split Node{" "}
                    </Menu.Item>
                  </MenuTooltip>
                  <Menu.Divider />
                  <Menu.Label>Misc</Menu.Label>
                  <MenuTooltip label="Make a comment about your flow.">
                    <Menu.Item onClick={addCommentNode} icon={"✏️"}>
                      {" "}
                      Comment Node{" "}
                    </Menu.Item>
                  </MenuTooltip>
                  {IS_RUNNING_LOCALLY ? (
                    <MenuTooltip label="Specify directories to load as local packages, so they can be imported in your Python evaluator nodes (add to sys path).">
                      <Menu.Item
                        onClick={addScriptNode}
                        icon={<IconSettingsAutomation size="16px" />}
                      >
                        {" "}
                        Global Python Scripts{" "}
                      </Menu.Item>
                    </MenuTooltip>
                  ) : (
                    <></>
                  )}
                </Menu.Dropdown>
              </Menu>

              <Button
                loading={loading}
                disabled={handleDisableSave()}
                // disabled={isCurrenFileLocked}
                size="sm"
                variant="outline"
                compact
                bg="#eee"
                mr="xs"
                onClick={() => {
                  handleSaveDropdown();
                }}
                rightIcon={
                  <div
                    ref={saveRef}
                    onClick={(e) => {
                      e.stopPropagation();
                      setSaveAndCommitBtnOpen(!saveAndCommitBtnOpen);
                      setOpenMenu(false);
                      setOpenAddNode(false);
                    }}
                  >
                    <div ref={saveRef}>
                      <Menu
                        position="top-start"
                        opened={saveAndCommitBtnOpen}
                        transitionProps={{ transition: "pop" }}
                      >
                        <div>
                          <Menu.Target>
                            <div
                              style={{
                                display: "flex",
                                alignItems: "center",
                              }}
                            >
                              <Chevron />
                            </div>
                          </Menu.Target>

                          <Menu.Dropdown>
                            <Menu.Item
                              onClick={() => {
                                setSaveDropdown("Save");
                                handleSaveFlow(false);
                              }}
                              style={{
                                backgroundColor:
                                  saveDropdown === "Save" ? "#e0e0e0" : "",
                              }}
                              disabled={isCurrenFileLocked}
                            >
                              Save
                            </Menu.Item>
                            <Menu.Item
                              onClick={() => {
                                setSaveDropdown("Save & Commit");
                                handleSaveAndCommit();
                              }}
                              style={{
                                backgroundColor:
                                  saveDropdown === "Save & Commit"
                                    ? "#e0e0e0"
                                    : "",
                              }}
                              disabled={isCurrenFileLocked}
                            >
                              Save & Commit
                            </Menu.Item>
                            <Menu.Item
                              onClick={() => {
                                setSaveDropdown("Deploy in app");
                              }}
                              style={{
                                backgroundColor:
                                  saveDropdown === "Deploy in app"
                                    ? "#e0e0e0"
                                    : "",
                              }}
                              disabled={!isCurrenFileLocked}
                            >
                              Deploy in app
                            </Menu.Item>
                          </Menu.Dropdown>
                        </div>
                      </Menu>
                    </div>
                  </div>
                }
              >
                {saveDropdown}
              </Button>
              {isCurrenFileLocked && (
                <span
                  style={{
                    fontSize: "12px",
                    marginLeft: "12px",
                    fontWeight: 400,
                    color: "#228be6",
                  }}
                ></span>
              )}
            </div>
            {activeUseCase.iteration && (
              <div className="center-tab">
                <div className="top-bar-usecase">
                  {activeUseCase.usecase &&
                    activeUseCase.usecase.split("__")[0]}
                </div>{" "}
                <div
                  className={`top-bar-iteration ${isCurrenFileLocked ? "committed" : ""}`}
                >
                  / {activeUseCase.iteration}
                </div>
                {isCurrenFileLocked && (
                  <div className="lock-icon">
                    <LockIcon />
                  </div>
                )}
              </div>
            )}
            <div
            // style={{ position: "fixed", right: "10px", top: "18px", zIndex: 8 }}
            >
              {IS_RUNNING_LOCALLY ? (
                <></>
              ) : (
                <Button
                  onClick={onClickShareFlow}
                  size="sm"
                  variant="outline"
                  compact
                  color={clipboard.copied ? "teal" : "blue"}
                  mr="xs"
                  style={{ float: "left" }}
                >
                  {waitingForShare ? (
                    <Loader size="xs" mr="4px" />
                  ) : (
                    <IconFileSymlink size="16px" />
                  )}
                  {clipboard.copied
                    ? "Link copied!"
                    : waitingForShare
                      ? "Sharing..."
                      : "Share"}
                </Button>
              )}
              <Button
                onClick={exportFlow}
                size="sm"
                variant="outline"
                bg="#eee"
                compact
                mr="xs"
                style={{ float: "left" }}
              >
                Export
              </Button>
              <Button
                onClick={importFlowFromFile}
                size="sm"
                variant="outline"
                bg="#eee"
                compact
                style={{ float: "left", marginRight: "8px" }}
              >
                Import
              </Button>
              <Menu
                transitionProps={{ transition: "pop-top-left" }}
                position="bottom-end"
                closeOnClickOutside={true}
                closeOnEscape
              >
                <Menu.Target>
                  <Button
                    size="sm"
                    variant="outline"
                    bg="#eee"
                    compact
                    style={{ float: "left", marginRight: "8px" }}
                  >
                    <Chevron />
                  </Button>
                </Menu.Target>

                <Menu.Dropdown>
                  <Menu.Item onClick={onClickExamples}>Examples</Menu.Item>
                </Menu.Dropdown>
              </Menu>
              <Button
                onClick={onClickSettings}
                size="sm"
                variant="gradient"
                compact
              >
                <IconSettings size={"90%"} />
              </Button>
            </div>
          </div>
        </Tooltip>
        <div
          style={{
            position: "fixed",
            right: "10px",
            bottom: "20px",
            zIndex: 8,
          }}
        >
          <a
            href="https://github.com/genai-apps/aggrag/issues/new/choose"
            target="_blank"
            style={{ color: "#666", fontSize: "11pt" }}
            rel="noreferrer"
          >
            Suggest a feature / raise a bug
          </a>
        </div>
      </div>
    );
};

export default App;
