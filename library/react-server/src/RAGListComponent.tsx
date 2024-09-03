import React, {
  useState,
  useEffect,
  useCallback,
  useRef,
  forwardRef,
  useImperativeHandle,
  useReducer,
  useMemo,
} from "react";
import {
  DragDropContext,
  Draggable,
  DraggableProvided,
  DraggableRubric,
  DraggableStateSnapshot,
  DroppableProvided,
  OnDragEndResponder,
} from "react-beautiful-dnd";
import { v4 as uuid } from "uuid";
import RAGListItem, { RAGListItemClone } from "./RAGListItem";
import { StrictModeDroppable } from "./StrictModeDroppable";
import ModelSettingsModal, {
  ModelSettingsModalRef,
} from "./ModelSettingsModal";
import {
  getDefaultModelFormData,
  getDefaultModelSettings,
} from "./ModelSettingSchemas";
import useStore, { initRAGProviderMenu } from "./store";
import { Dict, JSONCompatible, LLMSpec } from "./backend/typing";
import { useContextMenu } from "mantine-contextmenu";
import { ContextMenuItemOptions } from "mantine-contextmenu/dist/types";
import { Tooltip } from "@mantine/core";

// The RAG(s) to include by default on a PromptNode whenever one is created.
// Helper funcs
// Ensure that a name is 'unique'; if not, return an amended version with a count tacked on (e.g. "GPT-4 (2)")
const ensureUniqueName = (_name: string, _prev_names: string[]) => {
  // Strip whitespace around names
  const prev_names = _prev_names.map((n) => n.trim());
  const name = _name.trim();

  // Check if name is unique
  if (!prev_names.includes(name)) return name;

  // Name isn't unique; find a unique one:
  let i = 2;
  let new_name = `${name} (${i})`;
  while (prev_names.includes(new_name)) {
    i += 1;
    new_name = `${name} (${i})`;
  }
  return new_name;
};

/** Get position CSS style below and left-aligned to the input element */
const getPositionCSSStyle = (
  elem: HTMLButtonElement,
): ContextMenuItemOptions => {
  const rect = elem.getBoundingClientRect();
  return {
    key: "contextmenu",
    style: {
      position: "absolute",
      left: `${rect.left}px`,
      top: `${rect.bottom}px`,
    },
  };
};

export function RAGList({
  rags,
  onItemsChange,
  hideTrashIcon,
}: {
  rags: LLMSpec[];
  onItemsChange: (new_items: LLMSpec[]) => void;
  hideTrashIcon: boolean;
}) {
  const [items, setItems] = useState(rags);
  const settingsModal = useRef<ModelSettingsModalRef>(null);
  const [selectedModel, setSelectedModel] = useState<LLMSpec | undefined>(
    undefined,
  );

  const updateItems = useCallback(
    (new_items: LLMSpec[]) => {
      setItems(new_items);
      onItemsChange(new_items);
    },
    [onItemsChange],
  );

  const onClickSettings = useCallback(
    (item: LLMSpec) => {
      if (settingsModal && settingsModal.current) {
        setSelectedModel(item);
        settingsModal.current.trigger();
      }
    },
    [settingsModal],
  );

  const onSettingsSubmit = useCallback(
    (
      savedItem: LLMSpec,
      formData: Dict<JSONCompatible>,
      settingsData: Dict<JSONCompatible>,
    ) => {
      // First check for the item with key and get it:
      const rag = items.find((i) => i.key === savedItem.key);
      if (!rag) {
        console.error(
          `Could not update model settings: Could not find item with key ${savedItem.key}.`,
        );
        return;
      }

      const prev_names = items
        .filter((item) => item.key !== savedItem.key)
        .map((item) => item.name);

      // Change the settings for the RAG item to the value of 'formData':
      updateItems(
        items.map((item) => {
          if (item.key === savedItem.key) {
            // Create a new item with the same settings
            const updated_item: LLMSpec = { ...item };
            updated_item.formData = { ...formData };
            updated_item.settings = { ...settingsData };

            if ("model" in formData) {
              // Update the name of the specific model to call
              if (item.base_model.startsWith("__custom"))
                // Custom models must always have their base name, to avoid name collisions
                updated_item.model = item.base_model + "/" + formData.model;
              else updated_item.model = formData.model as string;
            }
            if ("shortname" in formData) {
              // Change the name, amending any name that isn't unique to ensure it is unique:
              const unique_name = ensureUniqueName(
                formData.shortname as string,
                prev_names,
              );
              updated_item.name = unique_name;
              if (updated_item.formData?.shortname)
                updated_item.formData.shortname = unique_name;
              if (updated_item.formData?.index_name) {
                if (unique_name.includes(" ")) {
                  const uid = unique_name
                    .toLowerCase()
                    ?.replace(" ", "_index_")
                    ?.replace(/[{()}]/g, "");
                  updated_item.formData.index_name = uid;
                } else {
                  updated_item.formData.index_name =
                    unique_name.toLowerCase() + "_index_1";
                }
                updated_item.settings.index_name =
                  updated_item.formData.index_name;
              }
            }

            if (savedItem.emoji) updated_item.emoji = savedItem.emoji;

            return updated_item;
          } else return item;
        }),
      );
    },
    [items, updateItems],
  );

  const onDragEnd: OnDragEndResponder = (result) => {
    const { destination, source } = result;
    if (!destination) return;
    if (
      (destination.droppableId === source.droppableId &&
        destination.index === source.index) ||
      !result.destination
    ) {
      return;
    }
    const newItems = Array.from(items);
    const [removed] = newItems.splice(result.source.index, 1);
    newItems.splice(result.destination.index, 0, removed);
    setItems(newItems);
  };

  const removeItem = useCallback(
    (item_key: string) => {
      // Double-check that the item we want to remove is in the list of items...
      if (!items.find((i) => i.key === item_key)) {
        console.error(
          `Could not remove model from RAG list: Could not find item with key ${item_key}.`,
        );
        return;
      }
      // Remove it
      updateItems(items.filter((i) => i.key !== item_key));
    },
    [items, updateItems],
  );

  useEffect(() => {
    // When RAGs list changes, we need to add new items
    // while preserving the current order of 'items'.
    // Check for new items and for each, add to end:
    const new_items = Array.from(
      items ? items.filter((i) => rags.some((v) => v.key === i.key)) : [],
    );
    if (rags) {
      rags.forEach((item) => {
        if (!items.find((i) => i.key === item?.key)) new_items.push(item);
      });
    }
    updateItems(new_items);
  }, [rags]);

  return (
    <div className="list nowheel nodrag">
      <ModelSettingsModal
        ref={settingsModal}
        model={selectedModel}
        onSettingsSubmit={onSettingsSubmit}
      />
      <DragDropContext onDragEnd={onDragEnd}>
        <StrictModeDroppable
          droppableId="rag-list-droppable"
          renderClone={(
            provided: DraggableProvided,
            snapshot: DraggableStateSnapshot,
            rubric: DraggableRubric,
          ) => (
            <RAGListItemClone
              provided={provided}
              snapshot={snapshot}
              item={items[rubric.source.index]}
              hideTrashIcon={hideTrashIcon}
            />
          )}
        >
          {(provided: DroppableProvided) => (
            <div {...provided.droppableProps} ref={provided.innerRef}>
              {items.map((item, index) => (
                <Draggable
                  key={item.key}
                  draggableId={item.key ?? index.toString()}
                  index={index}
                >
                  {(provided, snapshot) => (
                    <RAGListItem
                      provided={provided}
                      snapshot={snapshot}
                      item={item}
                      removeCallback={removeItem}
                      progress={item.progress}
                      onClickSettings={() => onClickSettings(item)}
                      hideTrashIcon={hideTrashIcon}
                    />
                  )}
                </Draggable>
              ))}
              {provided.placeholder}
            </div>
          )}
        </StrictModeDroppable>
      </DragDropContext>
    </div>
  );
}

export interface RAGListContainerRef {
  resetRAGItemsProgress: () => void;
  setZeroPercProgress: () => void;
  updateProgress: (itemProcessorFunc: (rag: LLMSpec) => LLMSpec) => void;
  ensureRAGItemsErrorProgress: (rag_keys_w_errors: string[]) => void;
  getRAGListItemForKey: (key: string) => LLMSpec | undefined;
  refreshRAGProviderList: () => void;
  setCreateIndexBtnDisabled: () => void;
}

export interface RAGListContainerProps {
  initRAGItems: LLMSpec[];
  description?: string;
  modelSelectButtonText?: string;
  onIndexBtnClick: () => void;
  onSelectModel?: (rag: LLMSpec, new_rags: LLMSpec[]) => void;
  onItemsChange?: (new_rags: LLMSpec[], old_rags: LLMSpec[]) => void;
  hideTrashIcon?: boolean;
  bgColor?: string;
  selectModelAction?: "add" | "replace";
}

export const RAGListContainer = forwardRef<
  RAGListContainerRef,
  RAGListContainerProps
>(function RAGListContainer(
  {
    description,
    modelSelectButtonText,
    initRAGItems,
    onIndexBtnClick,
    onSelectModel,
    selectModelAction,
    onItemsChange,
    hideTrashIcon,
    bgColor,
  },
  ref,
) {
  // All available RAG providers, for the dropdown list
  const AvailableRAGs = useStore((state) => state.AvailableRAGs);
  const { showContextMenu, hideContextMenu, isContextMenuVisible } =
    useContextMenu();
  // For some reason, when the AvailableRAGs list is updated in the store/, it is not
  // immediately updated here. I've tried all kinds of things, but cannot seem to fix this problem.
  // We must force a re-render of the component:
  // eslint-disable-next-line
  const [ignored, forceUpdate] = useReducer((x) => x + 1, 0);
  const refreshRAGProviderList = () => {
    forceUpdate();
  };

  const [disable, setdisable] = useState(true);
  const setCreateIndexBtnDisabled = () => {
    setdisable(false);
  };

  // Selecting RAG models to prompt
  const [ragItems, setRAGItems] = useState(initRAGItems);
  const [ragItemsCurrState, setRAGItemsCurrState] = useState<LLMSpec[]>([]);
  const resetRAGItemsProgress = useCallback(() => {
    setRAGItems(
      ragItemsCurrState.map((item) => {
        item.progress = undefined;
        return item;
      }),
    );
  }, [ragItemsCurrState]);
  const setZeroPercProgress = useCallback(() => {
    setRAGItems(
      ragItemsCurrState.map((item) => {
        item.progress = { success: 0, error: 0 };
        return item;
      }),
    );
  }, [ragItemsCurrState]);
  const updateProgress = useCallback(
    (itemProcessorFunc: (rag: LLMSpec) => LLMSpec) => {
      setRAGItems(ragItemsCurrState.map(itemProcessorFunc));
    },
    [ragItemsCurrState],
  );
  const ensureRAGItemsErrorProgress = useCallback(
    (rag_keys_w_errors: string[]) => {
      setRAGItems(
        ragItemsCurrState.map((item) => {
          if (item.key !== undefined && rag_keys_w_errors.includes(item.key)) {
            if (!item.progress) item.progress = { success: 0, error: 100 };
            else {
              const succ_perc = item.progress.success;
              item.progress = { success: succ_perc, error: 100 - succ_perc };
            }
          } else {
            if (item.progress && item.progress.success === 0)
              item.progress = undefined;
          }

          return item;
        }),
      );
    },
    [ragItemsCurrState],
  );

  const getRAGListItemForKey = useCallback(
    (key: string) => {
      return ragItemsCurrState.find((item) => item.key === key);
    },
    [ragItemsCurrState],
  );

  const handleSelectModel = useCallback(
    (item: LLMSpec) => {
      // Give it a uid as a unique key (this is needed for the draggable list to support multiple same-model items; keys must be unique)
      item = { key: uuid(), ...item };

      // Generate the default settings for this model
      item.settings = getDefaultModelSettings(item.base_model);

      // Repair names to ensure they are unique
      const unique_name = ensureUniqueName(
        item.name,
        ragItemsCurrState.map((i) => i.name),
      );
      item.name = unique_name;
      if (unique_name.includes(" ")) {
        const uid = unique_name
          .toLowerCase()
          ?.replace(" ", "_index_")
          ?.replace(/[{()}]/g, "");
        item.formData = { shortname: unique_name, index_name: uid };
      } else {
        item.formData = {
          shortname: unique_name,
          index_name: unique_name.toLowerCase() + "_index_1",
        };
      }
      item.settings = {
        ...item.settings,
        index_name: item.formData.index_name,
      };

      let new_items: LLMSpec[] = [];
      if (selectModelAction === "add" || selectModelAction === undefined) {
        // Add model to the RAG list (regardless of it's present already or not).
        new_items = ragItemsCurrState.concat([item]);
      } else if (selectModelAction === "replace") {
        // Remove existing model from RAG list and replace with new one:
        new_items = [item];
      }

      setRAGItems(new_items);
      if (onSelectModel) onSelectModel(item, new_items);
    },
    [ragItemsCurrState, onSelectModel, selectModelAction, AvailableRAGs],
  );

  const onRAGListItemsChange = useCallback(
    (new_items: LLMSpec[]) => {
      setRAGItemsCurrState(new_items);
      if (onItemsChange) onItemsChange(new_items, ragItemsCurrState);
    },
    [setRAGItemsCurrState, onItemsChange],
  );

  // This gives the parent access to triggering methods on this object
  useImperativeHandle(ref, () => ({
    resetRAGItemsProgress,
    setZeroPercProgress,
    updateProgress,
    ensureRAGItemsErrorProgress,
    getRAGListItemForKey,
    refreshRAGProviderList,
    setCreateIndexBtnDisabled,
  }));

  const _bgStyle = useMemo(
    () => (bgColor ? { backgroundColor: bgColor } : {}),
    [bgColor],
  );

  const menuItems = useMemo(() => {
    const initModels: Set<string> = new Set<string>();
    const convert = (item: LLMSpec): ContextMenuItemOptions => {
      initModels.add(item.base_model);
      return {
        key: item.model,
        title: (
          <Tooltip
            label={item.description}
            withinPortal
            withArrow
            offset={30}
            position="right"
            zIndex={10000000}
            styles={{
              tooltip: {
                backgroundColor: "#212529",
                color: "#fff",
              },
            }}
          >
            <span>
              {item.emoji} {item.name}
            </span>
          </Tooltip>
        ),
        onClick: () => handleSelectModel(item),
      };
    };
    const res = initRAGProviderMenu.map(convert);

    for (const item of AvailableRAGs) {
      if (initModels.has(item.base_model)) {
        continue;
      }
      res.push({
        key: item.base_model,
        title: `${item.emoji} ${item.name}`,
        onClick: () => handleSelectModel(item),
      });
    }
    return res;
  }, [AvailableRAGs, handleSelectModel]);

  // Mantine ContextMenu does not fix the position of the menu
  // to be below the clicked button, so we must do it ourselves.
  const addBtnRef = useRef(null);
  const indexBtnRef = useRef(null);
  const [wasContextMenuToggled, setWasContextMenuToggled] = useState(false);

  return (
    <div className="llm-list-container nowheel" style={_bgStyle}>
      <div className="llm-list-backdrop" style={_bgStyle}>
        {description || "RAGs to query:"}
        <div
          className={`add-llm-model-btn ${!disable ? "create-index-btn" : ""} nodrag`}
        >
          <button
            ref={indexBtnRef}
            style={{ ..._bgStyle, cursor: disable ? "not-allowed" : "pointer" }}
            onClick={() => {
              onIndexBtnClick();
            }}
            disabled={disable}
          >
            {modelSelectButtonText ?? "Create Index"}
          </button>
        </div>
        <div className="add-llm-model-btn nodrag">
          <button
            ref={addBtnRef}
            style={_bgStyle}
            onPointerDownCapture={() => {
              setWasContextMenuToggled(
                isContextMenuVisible && wasContextMenuToggled,
              );
            }}
            onClick={(evt) => {
              if (wasContextMenuToggled) {
                setWasContextMenuToggled(false);
                return; // abort
              }
              // This is a hack ---without hiding, the context menu position is not always updated.
              // This is the case even if hideContextMenu() was triggered elsewhere.
              hideContextMenu();
              // Now show the context menu below the button:
              showContextMenu(
                menuItems,
                addBtnRef?.current
                  ? getPositionCSSStyle(addBtnRef.current)
                  : undefined,
              )(evt);

              // Save whether the context menu was open, before
              // onPointerDown in App.tsx could auto-close the menu.
              setWasContextMenuToggled(true);
            }}
          >
            {modelSelectButtonText ?? "Add +"}
          </button>
        </div>
      </div>
      <div className="nodrag">
        <RAGList
          rags={ragItems}
          onItemsChange={onRAGListItemsChange}
          hideTrashIcon={hideTrashIcon ?? false}
        />
      </div>
    </div>
  );
});
