import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { OllamaEmbeddings } from "@langchain/ollama";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";

const loader = new PDFLoader("./nke-10k-2023.pdf");
const docs = await loader.load();

const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});
const allSplits = await textSplitter.splitDocuments(docs);

const embeddings = new OllamaEmbeddings({
  model: "nomic-embed-text:latest",
});

const vectorStore = await MemoryVectorStore.fromDocuments(
  allSplits,
  embeddings,
);

const results = await vectorStore.similaritySearch(
  "When was Nike incorporated?",
  4,
);

console.log(results[0].pageContent);
