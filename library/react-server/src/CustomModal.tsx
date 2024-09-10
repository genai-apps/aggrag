import React from "react";
import { Modal, Box, Text, Button, Flex } from "@mantine/core";

interface CustomModalProps {
  titleText: string;
  opened: boolean;
  onClose: () => void;
  onConfirm: () => void;
  content: React.ReactNode;
  confirmDisabled?: boolean;
  loading?: boolean;
  styles?: object;
}

const CustomModal: React.FC<CustomModalProps> = ({
  titleText,
  opened,
  onClose,
  onConfirm,
  content,
  confirmDisabled = false,
  loading = false,
  styles = {},
}) => (
  <Modal
    opened={opened}
    onClose={onClose}
    title={<div style={{ fontWeight: 500 }}>{titleText}</div>}
    styles={{
      header: {
        backgroundColor:
          titleText.indexOf("Delete") > -1 ? "#ffa500" : "#228be6",
        color: "white",
      },
      root: { position: "relative", left: "-5%" },
      close: {
        color: "#fff",
        "&:hover": { color: "black" },
      },
      ...styles,
    }}
  >
    <Box maw={400} mx="auto" mt="md" mb="md">
      {content}
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
        onClick={onClose}
        style={{ width: "40%" }}
      >
        Cancel
      </Button>
      <Button
        variant="filled"
        color="blue"
        onClick={onConfirm}
        disabled={confirmDisabled}
        loading={loading}
        style={{ width: "40%" }}
      >
        Confirm
      </Button>
    </Flex>
  </Modal>
);

export default CustomModal;
