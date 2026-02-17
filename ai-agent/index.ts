import "dotenv/config";
console.log(process.env.TAVILY_API_KEY);

import { createAgent } from "langchain";
import { ChatOllama } from "@langchain/ollama";
import { TavilySearch } from "@langchain/tavily";
import { Document } from "@langchain/core/documents";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
// import { RecursiveCharaterTextSpliter } from "@langchain/textsplitters";

const apiKey = process.env.TAVILY_API_KEY;

const docs = [
  new Document({
    pageContent:
      "Dogs are great companions, known for their loyalty and friendliness.",
    metadata: { source: "mammal-pets-doc" },
  }),

  new Document({
    pageContent:
      "Cats are great companions, known for their loyalty and friendliness.",
    metadata: { source: "mammal-pets-doc" },
  }),
];

const searchTool = new TavilySearch({
  maxResults: 1,
  topic: "news",
  tavilyApiKey: apiKey,
});

const ollama = new ChatOllama({
  model: "qwen3:8b",
  temperature: 0.7,
});

const agent = createAgent({
  model: ollama,
  tools: [searchTool],
});

const chat = async () => {
  try {
    const response = await agent.invoke({
      messages:
        "what is today's latest news in mumbai feb 16 202 give me summarized answer with 5 bullet points",
    });

    console.log(response);
  } catch (e) {
    console.log(e);
  }
};

chat();
