import path from "path";
import dotenv from "dotenv";

// Load environment variables from the root .env file
const envPath = path.resolve(__dirname, "../.env"); // Changed from "../../.env" to "../.env"
const result = dotenv.config({ path: envPath });

if (result.error) {
  console.error("Error loading environment variables:", result.error);
  console.error("Tried to load from path:", envPath);
  process.exit(1);
}

// Import app after environment variables are loaded so that environment variables loaded properly
// eslint-disable-next-line import/first
import app from "./backend/flow";

console.log(`Express port from env is: ${process.env.PORT_EXPRESS}`);
const port = process.env.PORT_EXPRESS || 3001;
app.listen(port, () => {
  console.log(`Server running on port ${port}`);
  console.log("Environment variables loaded from:", envPath);
  console.log(
    "FLASK BASE URL SET from process: ",
    process.env.REACT_APP_API_URL,
  );
});
