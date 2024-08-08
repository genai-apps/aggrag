/* eslint-disable */
const path = require("path");
const dotenv = require("dotenv");
const fs = require("fs");

const envPath = path.resolve(__dirname, "../../.env");

const result = dotenv.config({ path: envPath });

if (result.error) {
  throw result.error;
}

const apiUrl = result.parsed.REACT_APP_API_URL;

const envFileContent = `REACT_APP_API_URL=${apiUrl}`;

// Write to .env.local
fs.writeFileSync(path.resolve(__dirname, ".env"), envFileContent);

console.log("Environment variable REACT_APP_API_URL loaded from", envPath);
console.log("Environment variable written to .env.local");
