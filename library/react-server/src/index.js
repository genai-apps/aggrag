import React from "react";
import ReactDOM from "react-dom/client";
import "./index.css";
import App from "./App";
import reportWebVitals from "./reportWebVitals";
import { ContextMenuProvider } from "mantine-contextmenu";
import { AlertModalProvider } from "./AlertModal";
import { NotificationProvider } from "./Notification";

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
  <React.StrictMode>
    <AlertModalProvider>
      <NotificationProvider>
        <ContextMenuProvider>
          <App />
        </ContextMenuProvider>
      </NotificationProvider>
    </AlertModalProvider>
  </React.StrictMode>,
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
