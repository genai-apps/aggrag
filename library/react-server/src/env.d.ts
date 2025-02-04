// src/env.d.ts
declare namespace NodeJS {
  interface ProcessEnv {
    // Common environment variables
    REACT_APP_API_URL: string;
    PORT_EXPRESS: string;
    OPENAI_API_KEY: string;
    NODE_ENV: string;

    // Azure OpenAI variables
    AZURE_OPENAI_API_KEY?: string;
    AZURE_OPENAI_ENDPOINT?: string;
    AZURE_OPENAI_API_VERSION?: string;
    AZURE_OPENAI_DEPLOYMENT_NAME?: string;

    // Together AI variables
    TOGETHER_API_KEY?: string;

    // Other API keys
    ANTHROPIC_API_KEY?: string;
    GOOGLE_API_KEY?: string;
    MISTRAL_API_KEY?: string;

    // Optional: Add any other environment variables your application uses
    [key: string]: string | undefined;
  }
}
