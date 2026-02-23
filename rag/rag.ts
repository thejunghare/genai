import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { OllamaEmbeddings, ChatOllama } from "@langchain/ollama";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import {
  RunnablePassthrough,
  RunnableSequence,
} from "@langchain/core/runnables";

// 1. INDEXING
const loader = new PDFLoader("./nke-10k-2023.pdf");
const docs = await loader.load();

const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});
const allSplits = await textSplitter.splitDocuments(docs);

// 2. EMBED + STORE
const embeddings = new OllamaEmbeddings({ model: "nomic-embed-text" });
const vectorStore = await MemoryVectorStore.fromDocuments(
  allSplits,
  embeddings,
);
const retriever = vectorStore.asRetriever();

// 3. PROMPT
const prompt = ChatPromptTemplate.fromTemplate(`
You are an assistant for question-answering tasks.
Use only the following retrieved context to answer the question.
If you don't know the answer, just say you don't know.

Context: {context}

Question: {question}
`);

// 4. LLM
const llm = new ChatOllama({ model: "qwen3:8b", temperature: 0.1 });

// 5. CHAIN — manual but clean
const chain = RunnableSequence.from([
  {
    context: retriever.pipe((docs) =>
      docs.map((d) => d.pageContent).join("\n\n"),
    ),
    question: new RunnablePassthrough(),
  },
  prompt,
  llm,
  new StringOutputParser(),
]);

// 6. ASK
const answer = await chain.invoke("When was Nike incorporated?");
console.log(answer);
