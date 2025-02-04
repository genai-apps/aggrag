import React, { useState } from "react";
import html2canvas from "html2canvas";
import html2pdf from "html2pdf.js";
import { Menu, Button, Modal, Code, Stack, Text } from "@mantine/core";
import { IconChevronDown, IconDownload } from "@tabler/icons-react";
import { Prism } from "@mantine/prism";

interface RunFlowDropdownProps {
  onValidate: () => Promise<any>;
  onRun: () => Promise<any>;
  isRunning: boolean;
  // Add these new props
  className?: string;
  style?: React.CSSProperties;
  compact?: boolean;
  size?: string;
  icon?: React.ReactNode;
}

// interface RunFlowDropdownProps {
//   onValidate: () => Promise<any>;
//   onRun: () => Promise<any>;
//   isRunning: boolean;
// }

const RunFlowDropdown: React.FC<RunFlowDropdownProps> = ({
  onValidate,
  onRun,
  isRunning,
  className,
  style,
  compact,
  size,
  icon,
}) => {
  const [showValidationModal, setShowValidationModal] = useState(false);
  const [validationResult, setValidationResult] = useState<any>(null);
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [showRunModal, setShowRunModal] = useState(false);
  const [runResult, setRunResult] = useState<any>(null);

  const handleRun = async () => {
    try {
      setRunResult({
        status: "Running",
        output: "Executing flow...",
        logs: null,
      });
      setShowRunModal(true);
      setDropdownOpen(false);

      const result = await onRun();
      console.log("result:");
      console.log(result);
      // Check if result exists and has success flag
      if (result) {
        setRunResult({
          status: "Success",
          output: result.results, // Directly use the results object
          logs: null,
        });
      } else {
        throw new Error("Invalid response from server");
      }
    } catch (error) {
      console.error("Run failed:", error);
      setRunResult({
        status: "Failed",
        output: error instanceof Error ? error.message : "Run failed",
        logs: null,
      });
    }
  };

  const handleValidate = async () => {
    try {
      const result = await onValidate();
      setValidationResult(result);
      setShowValidationModal(true);
      setDropdownOpen(false);
    } catch (error) {
      console.error("Validation failed:", error);
    }
  };

  const downloadAsPDF = async () => {
    try {
      const modalContent = document.querySelector(
        ".validation-modal-content",
      ) as HTMLElement;
      if (!modalContent) {
        console.error("Modal content not found");
        return;
      }

      // Create a clone of the modal content
      const clone = modalContent.cloneNode(true) as HTMLElement;

      // Remove any max-height and overflow restrictions from the clone
      const scrollableElements = clone.querySelectorAll("*");
      scrollableElements.forEach((element: Element) => {
        if (element instanceof HTMLElement) {
          const computedStyle = window.getComputedStyle(element);
          if (
            computedStyle.overflow === "auto" ||
            computedStyle.overflow === "scroll" ||
            computedStyle.overflowY === "auto" ||
            computedStyle.overflowY === "scroll"
          ) {
            element.style.maxHeight = "none";
            element.style.overflow = "visible";
            element.style.overflowY = "visible";
          }
        }
      });

      // Create a wrapper with white background and proper styling
      const wrapper = document.createElement("div");
      wrapper.style.backgroundColor = "#ffffff";
      wrapper.style.padding = "20px";
      wrapper.appendChild(clone);

      // Calculate the full height including scrollable content
      document.body.appendChild(wrapper);
      const fullHeight = wrapper.scrollHeight;
      wrapper.style.height = `${fullHeight}px`;

      // Configure pdf options
      const opt = {
        margin: 10,
        filename: "validation-results.pdf",
        image: { type: "jpeg", quality: 0.98 },
        html2canvas: {
          scale: 2,
          useCORS: true,
          logging: false,
          height: fullHeight,
          windowHeight: fullHeight,
          onclone: (clonedDoc: Document) => {
            const elements = clonedDoc.getElementsByClassName(
              "validation-modal-content",
            );
            for (let i = 0; i < elements.length; i++) {
              const element = elements[i] as HTMLElement;
              element.style.display = "block";
              element.style.maxHeight = "none";
              element.style.overflow = "visible";
            }
          },
        },
        jsPDF: {
          unit: "mm",
          format: "a4",
          orientation: "portrait",
        },
      };

      try {
        // Generate PDF
        await html2pdf().set(opt).from(wrapper).save();
      } finally {
        // Cleanup
        document.body.removeChild(wrapper);
      }
    } catch (error) {
      console.error("Failed to generate PDF:", error);
    }
  };

  return (
    <>
      <Menu opened={dropdownOpen}>
        <Menu.Target>
          <Button
            className={className}
            style={{
              ...style,
              marginRight: "8px",
            }}
            sx={{
              "&:hover": {
                backgroundColor: "#3dd409",
              },
            }}
            compact={compact}
            size={size}
            disabled={isRunning}
            rightIcon={
              <div
                onClick={(e) => {
                  e.stopPropagation();
                  setDropdownOpen(!dropdownOpen);
                }}
              >
                <IconChevronDown size="1rem" />
              </div>
            }
            onClick={handleValidate}
          >
            {icon}
            {isRunning ? "Running..." : "Validate Flow"}
          </Button>
        </Menu.Target>

        <Menu.Dropdown>
          <Menu.Item onClick={handleValidate}>Validate Flow</Menu.Item>
          <Menu.Item
            onClick={() => {
              handleRun();
              setDropdownOpen(false);
            }}
          >
            Run Flow
          </Menu.Item>
        </Menu.Dropdown>
      </Menu>
      <Modal
        opened={showRunModal}
        onClose={() => setShowRunModal(false)}
        size="lg"
        title={
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              width: "100%",
            }}
          >
            <Text>Flow Run Results</Text>
          </div>
        }
        styles={{
          content: {
            minHeight: "200px",
            position: "relative",
          },
          body: {
            padding: "20px",
          },
        }}
        classNames={{
          content: "validation-modal-content",
        }}
      >
        {runResult && (
          <Stack spacing="md">
            <Text weight={500}>Run Status:</Text>
            <Code
              block
              color={
                runResult.status === "Success"
                  ? "green"
                  : runResult.status === "Running"
                    ? "yellow"
                    : "red"
              }
            >
              {runResult.status}
            </Code>

            <Text weight={500}>Results:</Text>
            <Prism language="json" withLineNumbers>
              {typeof runResult.output === "string"
                ? runResult.output || "No output"
                : JSON.stringify(runResult.output || {}, null, 2)}
            </Prism>

            {runResult.logs && (
              <>
                <Text weight={500}>Execution Logs:</Text>
                <Code block style={{ maxHeight: "300px", overflow: "auto" }}>
                  {typeof runResult.logs === "string"
                    ? runResult.logs
                    : JSON.stringify(runResult.logs, null, 2)}
                </Code>
              </>
            )}
          </Stack>
        )}
      </Modal>
      <Modal
        opened={showValidationModal}
        onClose={() => setShowValidationModal(false)}
        size="lg"
        title={
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              width: "100%",
            }}
          >
            <Text>Flow Validation Results</Text>
            <Button
              variant="subtle"
              size="sm"
              leftIcon={<IconDownload size="1rem" />}
              onClick={downloadAsPDF}
            ></Button>
          </div>
        }
        styles={{
          content: {
            minHeight: "200px", // Ensure minimum height
            position: "relative",
          },
          body: {
            padding: "20px",
          },
        }}
        classNames={{
          content: "validation-modal-content",
        }}
      >
        {validationResult && (
          <Stack>
            <Text weight={500}>Execution Order:</Text>
            {validationResult.validation.executionOrder.map(
              (level: any[], index: number) => (
                <div key={index}>
                  <Text size="sm">Level {index + 1} (Parallel Execution):</Text>
                  <ul>
                    {level.map((node) => (
                      <li key={node.id}>
                        {node.name} ({node.type})
                      </li>
                    ))}
                  </ul>
                </div>
              ),
            )}

            <Text weight={500}>Required Variables In API Request:</Text>
            <ul>
              {validationResult.validation.requiredVariables.map(
                (variable: string) => (
                  <li key={variable}>@{variable}</li>
                ),
              )}
            </ul>

            <Text weight={500}> Sample API Request To Run Flow (curl):</Text>
            <Prism language="bash">
              {validationResult.validation.curlExample}
            </Prism>
          </Stack>
        )}
      </Modal>
    </>
  );
};

export default RunFlowDropdown;
