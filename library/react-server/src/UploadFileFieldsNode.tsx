import React, {
  useState,
  useRef,
  useEffect,
  useCallback,
  useMemo,
  MouseEventHandler,
} from "react";
import { Handle, Node, Position } from "reactflow";
import { Tooltip, Skeleton, Loader, Grid, Text } from "@mantine/core";
import {
  IconTextPlus,
  IconEye,
  IconEyeOff,
  IconTransform,
} from "@tabler/icons-react";
import useStore from "./store";
import NodeLabel from "./NodeLabelComponent";
import TemplateHooks, {
  extractBracketedSubstrings,
} from "./TemplateHooksComponent";
import BaseNode from "./BaseNode";
import {
  DebounceRef,
  genDebounceFunc,
  upload_raw_docs_file,
  setsAreEqual,
} from "./backend/utils";
import { Func, Dict } from "./backend/typing";
import {
  ItemsNodeProps,
  makeSafeForCSLFormat,
  prepareItemsNodeData,
} from "./ItemsNode";
import { useNotification } from "./Notification";

// Helper funcs
const union = (setA: Set<any>, setB: Set<any>) => {
  const _union = new Set(setA);
  for (const elem of setB) {
    _union.add(elem);
  }
  return _union;
};

const delButtonId = "del-";
const visibleButtonId = "eye-";

interface FileFieldsNodeData {
  vars?: string[];
  title?: string;
  text?: string;
  fields?: Dict<string>;
  fields_visibility?: Dict<boolean>;
  refresh?: boolean;
}

export interface FileFieldsNodeProps {
  data: FileFieldsNodeData;
  id: string;
}

const FileFieldsNode: React.FC<FileFieldsNodeProps> = ({ data, id }) => {
  const [templateVars, setTemplateVars] = useState(data.vars || []);
  const duplicateNode = useStore((state) => state.duplicateNode);
  const addNode = useStore((state) => state.addNode);
  const removeNode = useStore((state) => state.removeNode);
  const setDataPropsForNode = useStore((state) => state.setDataPropsForNode);
  const pingOutputNodes = useStore((state) => state.pingOutputNodes);
  const { showNotification } = useNotification();
  const setTriggerHint = useStore((state) => state.setTriggerHint);
  const [filefieldsValues, setFilefieldsValues] = useState<Dict<string>>(
    data.fields ?? {},
  );
  const [fieldVisibility, setFieldVisibility] = useState<Dict<boolean>>(
    data.fields_visibility || {},
  );

  // Whether the file fields should be in a loading state
  const [isLoading, setIsLoading] = useState(false);

  // Debounce helpers
  const debounceTimeoutRef: DebounceRef = useRef(null);
  const debounce: Func<Func> = genDebounceFunc(debounceTimeoutRef);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const queryString = window.location.search;
  const urlParams = new URLSearchParams(queryString);

  const getUID = useCallback((fileFields: Dict<string>) => {
    if (fileFields) {
      return (
        "f" +
        (
          1 +
          Object.keys(fileFields).reduce(
            (acc, key) => Math.max(acc, parseInt(key.slice(1))),
            0,
          )
        ).toString()
      );
    } else {
      return "f0";
    }
  }, []);

  // Handle delete file field.
  const handleDelete: MouseEventHandler<HTMLButtonElement> = useCallback(
    (event: React.MouseEvent<HTMLButtonElement>) => {
      // Update the data for this file field's id.
      const new_fields = { ...filefieldsValues };
      const new_vis = { ...fieldVisibility };
      const item_id = (event.target as HTMLButtonElement).id.substring(
        delButtonId.length,
      );
      delete new_fields[item_id];
      delete new_vis[item_id];
      // if the new_data is empty, initialize it with one empty field
      if (Object.keys(new_fields).length === 0) {
        new_fields[getUID(filefieldsValues)] = "";
      }
      setFilefieldsValues(new_fields);
      setFieldVisibility(new_vis);
      setDataPropsForNode(id, {
        fields: new_fields,
        fields_visibility: new_vis,
      });
      pingOutputNodes(id);
    },
    [
      filefieldsValues,
      fieldVisibility,
      id,
      delButtonId,
      setDataPropsForNode,
      pingOutputNodes,
    ],
  );

  // Initialize fields (run once at init)
  useEffect(() => {
    if (!filefieldsValues || Object.keys(filefieldsValues).length === 0) {
      const init_fields: Dict<string> = {};
      init_fields[getUID(filefieldsValues)] = "";
      setFilefieldsValues(init_fields);
      setDataPropsForNode(id, { fields: init_fields });
    }
  }, []);

  // Add a file field
  const handleAddField = useCallback(() => {
    const new_fields = { ...filefieldsValues };
    new_fields[getUID(filefieldsValues)] = "";
    setFilefieldsValues(new_fields);
    setDataPropsForNode(id, { fields: new_fields });
    pingOutputNodes(id);

    // Cycle suggestions when new field is created
    // aiSuggestionsManager.cycleSuggestions();
  }, [filefieldsValues, id, setDataPropsForNode, pingOutputNodes]);

  // Disable/hide a file field temporarily
  const handleDisableField = useCallback(
    (field_id: string) => {
      const vis = { ...fieldVisibility };
      vis[field_id] = fieldVisibility[field_id] === false; // toggles it
      setFieldVisibility(vis);
      setDataPropsForNode(id, { fields_visibility: vis });
      pingOutputNodes(id);
    },
    [fieldVisibility, setDataPropsForNode, pingOutputNodes],
  );

  const updateTemplateVars = useCallback(
    (new_data: FileFieldsNodeData) => {
      // TODO: Optimize this check.
      let all_found_vars = new Set<string>();
      const new_field_ids = Object.keys(new_data.fields ?? {});
      new_field_ids.forEach((fid: string) => {
        const found_vars = extractBracketedSubstrings(
          (new_data.fields as Dict<string>)[fid],
        );
        if (found_vars && found_vars.length > 0) {
          all_found_vars = union(all_found_vars, new Set(found_vars));
        }
      });

      // Update template var fields + handles, if there's a change in sets
      const past_vars = new Set(templateVars);
      if (!setsAreEqual(all_found_vars, past_vars)) {
        const new_vars_arr = Array.from(all_found_vars);
        new_data.vars = new_vars_arr;
      }

      return new_data;
    },
    [templateVars],
  );

  // Save the state of a filefield when it changes and update hooks
  const handleFileFieldChange = useCallback(
    (field_id: string, val: string, shouldDebounce: boolean) => {
      // Update the value of the controlled component
      const new_fields = { ...filefieldsValues };
      new_fields[field_id] = val;
      setFilefieldsValues(new_fields);
      setIsLoading(false);

      // Update the data for the ReactFlow node
      const new_data = updateTemplateVars({ fields: new_fields });
      if (new_data.vars) setTemplateVars(new_data.vars);

      // Debounce the global state change to happen only after 300ms, as it forces a costly rerender:
      debounce(
        (_id: string, _new_data: FileFieldsNodeData) => {
          setDataPropsForNode(_id, _new_data as Dict);
          pingOutputNodes(_id);
        },
        shouldDebounce ? 300 : 1,
      )(id, new_data);
    },
    [
      filefieldsValues,
      setFilefieldsValues,
      templateVars,
      updateTemplateVars,
      setTemplateVars,
      pingOutputNodes,
      setDataPropsForNode,
      id,
    ],
  );

  async function handleFileUpload(event: any, field_id: string) {
    if (event.target.files && event.target.files[0]) {
      if (
        event.target.files[0].type === "application/pdf" ||
        event.target.files[0].type === "application/msword" ||
        event.target.files[0].type ===
          "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
      ) {
        const formData = new FormData();
        const timestamp = Date.now().toString();
        const fid = `fnid-${id.split("-")[1]}`;
        formData.append("file", event.target.files[0]);
        formData.append("timestamp", timestamp);
        formData.append("file_node_id", fid);
        formData.append("p_folder", urlParams.get("p_folder") || "");
        formData.append("i_folder", urlParams.get("i_folder") || "");
        try {
          setIsLoading(true);
          const response = await upload_raw_docs_file(formData);
          const file_path = `${urlParams.get("p_folder")}/${urlParams.get("i_folder")}/raw_docs/${fid}_${timestamp}/${event.target.files[0].name}`;
          handleFileFieldChange(field_id, file_path, false);
          setTriggerHint("file-upload");
        } catch (error) {
          event.target.value = null;
          showNotification("Failed", "Error uploading file", "red");
          setIsLoading(false);
        }
      }
    }
  }

  // Dynamically update the fileareas and position of the template hooks
  const ref = useRef<HTMLDivElement | null>(null);
  const [hooksY, setHooksY] = useState(120);
  useEffect(() => {
    const node_height = ref?.current?.clientHeight ?? 0;
    setHooksY(node_height + 68);
  }, [filefieldsValues]);

  const setRef = useCallback(
    (elem: HTMLDivElement) => {
      // To listen for resize events of the filearea, we need to use a ResizeObserver.
      // We initialize the ResizeObserver only once, when the 'ref' is first set, and only on the div wrapping filefields.
      // NOTE: This won't work on older browsers, but there's no alternative solution.

      if (!ref.current && elem && window.ResizeObserver) {
        let past_hooks_y = 120;
        const observer = new window.ResizeObserver(() => {
          if (!ref || !ref.current) return;
          const new_hooks_y = ref.current.clientHeight + 68;
          if (past_hooks_y !== new_hooks_y) {
            setHooksY(new_hooks_y);
            past_hooks_y = new_hooks_y;
          }
        });

        observer.observe(elem);
        ref.current = elem;
      }
    },
    [ref],
  );

  // Pass upstream changes down to later nodes in the chain
  const refresh = useMemo(() => data.refresh, [data.refresh]);
  useEffect(() => {
    if (refresh === true) {
      pingOutputNodes(id);
      setDataPropsForNode(id, { refresh: false });
    }
  }, [refresh]);

  function getFileName(filePath: string) {
    return filePath.split("/").pop();
  }

  // Cache the rendering of the file fields.
  const fileFields = useMemo(
    () =>
      Object.keys(filefieldsValues).map((i) => {
        return (
          <div className="input-field" key={i}>
            <label
              htmlFor={"pdfWordFile" + i}
              className="upload-file"
              style={{ width: "285px" }}
            >
              {filefieldsValues[i] ? (
                <Text
                  className="text-field-fixed nodrag nowheel"
                  truncate="end"
                >
                  {getFileName(filefieldsValues[i])}
                </Text>
              ) : (
                <input
                  className="text-field-fixed nodrag nowheel"
                  ref={fileInputRef}
                  formEncType="multipart/form-data"
                  type="file"
                  id={"pdfWordFile" + i}
                  name={"pdfWordFile" + i}
                  // value={filefieldsValues[i]}
                  disabled={fieldVisibility[i] === false}
                  accept="application/msword, application/vnd.ms-excel, application/vnd.ms-powerpoint, text/plain, application/pdf"
                  onChange={(event) => handleFileUpload(event, i)}
                />
              )}
            </label>
            {Object.keys(filefieldsValues).length > 1 ? (
              <div style={{ display: "flex", flexDirection: "column" }}>
                <Tooltip
                  label="remove field"
                  position="right"
                  withArrow
                  arrowSize={10}
                  withinPortal
                >
                  <button
                    id={delButtonId + i}
                    className="remove-text-field-btn nodrag"
                    onClick={handleDelete}
                    style={{ flex: 1 }}
                  >
                    X
                  </button>
                </Tooltip>
                <Tooltip
                  label={
                    (fieldVisibility[i] === false ? "enable" : "disable") +
                    " field"
                  }
                  position="right"
                  withArrow
                  arrowSize={10}
                  withinPortal
                >
                  <button
                    id={visibleButtonId + i}
                    className="remove-text-field-btn nodrag"
                    onClick={() => handleDisableField(i)}
                    style={{ flex: 1 }}
                  >
                    {fieldVisibility[i] === false ? (
                      <IconEyeOff size="14pt" pointerEvents="none" />
                    ) : (
                      <IconEye size="14pt" pointerEvents="none" />
                    )}
                  </button>
                </Tooltip>
              </div>
            ) : (
              <></>
            )}
          </div>
        );
      }),
    // Update the file fields only when their values, placeholders, or visibility changes.
    [filefieldsValues, fieldVisibility],
  );

  // Add custom context menu options on right-click.
  // 1. Convert FileFields to Items Node, for convenience.
  const customContextMenuItems = useMemo(
    () => [
      {
        key: "to_item_node",
        icon: <IconTransform size="11pt" />,
        text: "To Items Node",
        onClick: () => {
          // Convert the fields of this node into Items Node CSL format:
          const csl_fields =
            Object.values(filefieldsValues).map(makeSafeForCSLFormat);
          const text = csl_fields.join(", ");
          // Duplicate this FileFields node
          const dup = duplicateNode(id) as Node;
          // Swap the data for new data:
          const items_node_data: ItemsNodeProps["data"] = {
            title: dup.data.title,
            text,
            fields: prepareItemsNodeData(text).fields,
          };
          // Add the duplicated node, with correct type:
          addNode({
            ...dup,
            id: `csvNode-${Date.now()}`,
            type: `csv`,
            data: items_node_data,
          });
          // Remove the current TF node on redraw:
          removeNode(id);
        },
      },
    ],
    [id, filefieldsValues],
  );

  return (
    <BaseNode
      classNames={`file-fields-node ${id}`}
      nodeId={id}
      contextMenuExts={customContextMenuItems}
    >
      <NodeLabel
        title={data.title ?? "FileFields Node"}
        nodeId={id}
        icon={<IconTextPlus size="16px" />}
        customButtons={[]}
      />

      <Grid style={{ marginBottom: 0 }}>
        <Grid.Col span={isLoading ? 8 : 12}>
          <Skeleton visible={isLoading}>
            <div ref={setRef}>{fileFields}</div>
          </Skeleton>
        </Grid.Col>
        {isLoading && (
          <Grid.Col
            span={4}
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <Loader size={20} />
          </Grid.Col>
        )}
      </Grid>
      <Handle
        type="source"
        position={Position.Right}
        id="output"
        className="grouped-handle"
        style={{ top: "50%" }}
      />
      <TemplateHooks
        vars={templateVars}
        nodeId={id}
        startY={hooksY}
        position={Position.Left}
      />
      <div className="add-text-field-btn">
        <button onClick={handleAddField}>+</button>
      </div>
    </BaseNode>
  );
};

export default FileFieldsNode;
