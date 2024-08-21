import React, { createContext, useContext, useState, useCallback } from "react";
import { Notification, Transition } from "@mantine/core";
import { IconCheck, IconX } from "@tabler/icons-react";

interface NotificationContextType {
  showNotification: (title: string, text: string, color?: string) => void;
}

const NotificationContext = createContext<NotificationContextType | undefined>(
  undefined,
);

export const useNotification = () => {
  const context = useContext(NotificationContext);
  if (!context) {
    throw new Error(
      "useNotification must be used within a NotificationProvider",
    );
  }
  return context;
};

export const NotificationProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [notification, setNotification] = useState({
    title: "",
    text: "",
    color: "teal",
  });
  const [show, setShow] = useState(false);

  const showNotification = useCallback(
    (title: string, text: string, color = "teal") => {
      setNotification({ title, text, color });
      setShow(true);
      setTimeout(() => setShow(false), 5000);
    },
    [],
  );

  return (
    <NotificationContext.Provider value={{ showNotification }}>
      {children}
      <Transition
        mounted={show}
        transition="slide-right"
        duration={300}
        timingFunction="ease"
      >
        {(styles) => (
          <Notification
            color={notification.color}
            title={notification.title}
            style={{ ...styles, position: "absolute", zIndex: 10, bottom: 30 }}
            onClose={() => setShow(false)}
            icon={
              notification.color === "red" ? (
                <IconX />
              ) : (
                <IconCheck style={{ width: 20, height: 20 }} />
              )
            }
          >
            {notification.text}
          </Notification>
        )}
      </Transition>
    </NotificationContext.Provider>
  );
};
