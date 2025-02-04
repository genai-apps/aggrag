/* eslint-disable @typescript-eslint/no-var-requires */
const path = require("path"); // @typescript-eslint/no-var-requires
const dotenv = require("dotenv"); // @typescript-eslint/no-var-requires
const fs = require("fs"); // @typescript-eslint/no-var-requires

const envPath = path.resolve(__dirname, "../../.env");
const result = dotenv.config({ path: envPath });

if (result.error) {
  throw result.error;
}

// Get all environment variables from the parsed result
const envVars = result.parsed;

// Create env content with all variables
const envFileContent = Object.entries(envVars)
  .map(([key, value]) => `${key}=${value}`)
  .join("\n");

// Write to .env
fs.writeFileSync(path.resolve(__dirname, ".env"), envFileContent);

console.log("Environment variables loaded from", envPath);
console.log("Environment variables written to .env");
