import { v4 as uuid } from "uuid";
import path from "path";

interface PyodideWorkerResponse {
  id: string;
  results?: any;
  error?: string;
}

interface PyodideCallbacks {
  [key: string]: (data: Omit<PyodideWorkerResponse, "id">) => void;
}

let pyodideWorker: Worker | undefined;
const callbacks: PyodideCallbacks = {};

const workerPath = path.join(__dirname, "exec-py.worker.js");

const execPy = (script: string, context: Record<string, unknown> = {}) => {
  // Initialize the worker only when first called, to save on load times
  if (!pyodideWorker) {
    pyodideWorker = new Worker(workerPath);
    pyodideWorker.onmessage = (event: MessageEvent<PyodideWorkerResponse>) => {
      const { id, ...data } = event.data;
      const onSuccess = callbacks[id];
      delete callbacks[id];
      onSuccess(data);
    };
  }

  const id = uuid();

  // Execute the worker
  return new Promise<Omit<PyodideWorkerResponse, "id">>((onSuccess) => {
    callbacks[id] = onSuccess;
    pyodideWorker?.postMessage({
      ...context,
      python: script,
      id,
    });
  });
};

export { execPy };
