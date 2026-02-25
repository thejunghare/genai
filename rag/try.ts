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
import { Document as LangDoc } from "@langchain/core/documents";
import * as fs from "fs";
import * as path from "path";

// ─── CONFIG ────────────────────────────────────────────────────────────────
// Drop your PDF paths here — add as many as you want
const PDF_FILES = ["./DOC/doc1.pdf", "./DOC/doc2.pdf", "./DOC/doc3.pdf"];

const OUTPUT_FILE = "./ieee-output.docx";
const TOPIC = "Named Entity Recognition in Marathi Language";
// ───────────────────────────────────────────────────────────────────────────

async function loadMultiplePDFs(filePaths: string[]): Promise<LangDoc[]> {
  const allDocs: LangDoc[] = [];

  for (const filePath of filePaths) {
    if (!fs.existsSync(filePath)) {
      console.warn(`⚠️  Skipping ${filePath} — file not found`);
      continue;
    }
    console.log(`📄 Loading: ${filePath}`);
    const loader = new PDFLoader(filePath);
    const docs = await loader.load();

    // Tag each doc with its source file
    const taggedDocs = docs.map((doc) => ({
      ...doc,
      metadata: { ...doc.metadata, sourceFile: path.basename(filePath) },
    }));

    allDocs.push(...taggedDocs);
  }

  console.log(
    `✅ Loaded ${allDocs.length} pages from ${filePaths.length} files\n`,
  );
  return allDocs;
}

async function buildRAGChain(docs: LangDoc[]) {
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
  const allSplits = await textSplitter.splitDocuments(docs);
  console.log(`✂️  Split into ${allSplits.length} chunks\n`);

  const embeddings = new OllamaEmbeddings({ model: "nomic-embed-text" });
  const vectorStore = await MemoryVectorStore.fromDocuments(
    allSplits,
    embeddings,
  );
  const retriever = vectorStore.asRetriever({ k: 8 });

  const llm = new ChatOllama({
    model: "mistral",
    temperature: 0.1,
    numCtx: 16384, // 16k context — optimal for M4 Air 16GB
  });

  const prompt = ChatPromptTemplate.fromTemplate(`
You are a senior research assistant specializing in Natural Language Processing,
specifically Named Entity Recognition (NER) in Marathi and other Indian languages.
You are writing an IEEE-format academic paper on the topic: "${TOPIC}".

Use ONLY the provided context from research papers to write your response.
Be precise, factual, and formal in academic tone.
Do not hallucinate or add information not present in the context.

Context:
{context}

Task: {question}
  `);

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

  return chain;
}

async function generateIEEEDoc(introduction: string, literatureSurvey: string) {
  // Dynamic import for docx (CommonJS module)
  const {
    Document,
    Packer,
    Paragraph,
    TextRun,
    AlignmentType,
    HeadingLevel,
    BorderStyle,
    UnderlineType,
  } = await import("docx");

  const sources = PDF_FILES.map((f, i) => path.basename(f));

  const doc = new Document({
    styles: {
      default: {
        document: { run: { font: "Times New Roman", size: 20 } }, // 10pt = IEEE standard
      },
      paragraphStyles: [
        {
          id: "Heading1",
          name: "Heading 1",
          basedOn: "Normal",
          next: "Normal",
          run: { size: 22, bold: true, font: "Times New Roman" },
          paragraph: {
            spacing: { before: 240, after: 120 },
            outlineLevel: 0,
          },
        },
      ],
    },
    sections: [
      {
        properties: {
          page: {
            size: { width: 12240, height: 15840 },
            margin: { top: 1440, right: 1080, bottom: 1440, left: 1080 },
          },
        },
        children: [
          // ── TITLE ──
          new Paragraph({
            alignment: AlignmentType.CENTER,
            spacing: { before: 0, after: 240 },
            children: [
              new TextRun({
                text: "Named Entity Recognition in Marathi Language: A Systematic Literature Survey",
                bold: true,
                size: 28,
                font: "Times New Roman",
              }),
            ],
          }),

          // ── AUTHORS ──
          new Paragraph({
            alignment: AlignmentType.CENTER,
            spacing: { before: 0, after: 480 },
            children: [
              new TextRun({
                text: "Generated via RAG Pipeline · LangChain + Ollama",
                italics: true,
                size: 20,
                font: "Times New Roman",
              }),
            ],
          }),

          // ── ABSTRACT HEADING ──
          new Paragraph({
            alignment: AlignmentType.CENTER,
            spacing: { before: 0, after: 120 },
            children: [
              new TextRun({
                text: "Abstract",
                bold: true,
                italics: true,
                size: 20,
                font: "Times New Roman",
              }),
            ],
          }),

          // ── ABSTRACT BODY ──
          new Paragraph({
            alignment: AlignmentType.BOTH,
            spacing: { before: 0, after: 480 },
            children: [
              new TextRun({
                text: `This paper presents a systematic literature survey on Named Entity Recognition (NER) in the Marathi language. NER is a fundamental task in Natural Language Processing (NLP) that identifies and classifies named entities such as persons, locations, and organizations in text. Marathi, a morphologically rich Indic language spoken by over 83 million people, poses unique challenges for NER due to its complex morphology, limited annotated corpora, and code-mixing phenomena. This survey synthesizes findings from ${sources.length} research paper(s) to provide a comprehensive overview of existing methodologies, datasets, and future directions in Marathi NER.`,
                size: 20,
                font: "Times New Roman",
              }),
            ],
          }),

          // ── SECTION I: INTRODUCTION ──
          new Paragraph({
            heading: HeadingLevel.HEADING_1,
            spacing: { before: 240, after: 120 },
            children: [
              new TextRun({
                text: "I. Introduction",
                bold: true,
                size: 22,
                font: "Times New Roman",
              }),
            ],
          }),

          // Split intro into paragraphs
          ...introduction
            .split("\n")
            .filter((p) => p.trim().length > 0)
            .map(
              (para) =>
                new Paragraph({
                  alignment: AlignmentType.BOTH,
                  spacing: { before: 0, after: 120 },
                  indent: { firstLine: 720 },
                  children: [
                    new TextRun({
                      text: para.trim(),
                      size: 20,
                      font: "Times New Roman",
                    }),
                  ],
                }),
            ),

          // ── SECTION II: LITERATURE SURVEY ──
          new Paragraph({
            heading: HeadingLevel.HEADING_1,
            spacing: { before: 240, after: 120 },
            children: [
              new TextRun({
                text: "II. Literature Survey",
                bold: true,
                size: 22,
                font: "Times New Roman",
              }),
            ],
          }),

          ...literatureSurvey
            .split("\n")
            .filter((p) => p.trim().length > 0)
            .map(
              (para) =>
                new Paragraph({
                  alignment: AlignmentType.BOTH,
                  spacing: { before: 0, after: 120 },
                  indent: { firstLine: 720 },
                  children: [
                    new TextRun({
                      text: para.trim(),
                      size: 20,
                      font: "Times New Roman",
                    }),
                  ],
                }),
            ),

          // ── REFERENCES ──
          new Paragraph({
            heading: HeadingLevel.HEADING_1,
            spacing: { before: 240, after: 120 },
            children: [
              new TextRun({
                text: "References",
                bold: true,
                size: 22,
                font: "Times New Roman",
              }),
            ],
          }),

          ...sources.map(
            (src, i) =>
              new Paragraph({
                alignment: AlignmentType.BOTH,
                spacing: { before: 0, after: 80 },
                indent: { left: 720, hanging: 720 },
                children: [
                  new TextRun({
                    text: `[${i + 1}] `,
                    bold: true,
                    size: 20,
                    font: "Times New Roman",
                  }),
                  new TextRun({
                    text: `Source document: "${src}". Processed via RAG pipeline, ${new Date().getFullYear()}.`,
                    size: 20,
                    font: "Times New Roman",
                  }),
                ],
              }),
          ),
        ],
      },
    ],
  });

  const buffer = await Packer.toBuffer(doc);
  fs.writeFileSync(OUTPUT_FILE, buffer);
  console.log(`\n📝 IEEE document saved to: ${OUTPUT_FILE}`);
}

// ─── MAIN ──────────────────────────────────────────────────────────────────
async function main() {
  console.log("🚀 Starting RAG pipeline...\n");

  const docs = await loadMultiplePDFs(PDF_FILES);
  if (docs.length === 0) {
    console.error("❌ No documents loaded. Check your PDF_FILES paths.");
    process.exit(1);
  }

  const chain = await buildRAGChain(docs);

  console.log("🧠 Generating Introduction...");
  const introduction = await chain.invoke(
    `Write a detailed IEEE academic introduction section for a paper on "${TOPIC}".
    Cover the following points using information from the context:
    - What is Named Entity Recognition (NER) and its importance in NLP
    - Why Marathi language is significant and challenging for NER
    - Challenges specific to Marathi: morphological richness, lack of resources, code-mixing
    - Motivation and objectives of this survey
    - Brief overview of what the paper covers
    Write in formal IEEE academic style. Do not use bullet points — use flowing paragraphs.`,
  );

  console.log("📚 Generating Literature Survey...");
  const literatureSurvey = await chain.invoke(
    `Write a detailed IEEE literature survey section for a paper on "${TOPIC}".
    For EACH paper found in the context, extract and present the following information in this EXACT structured format:

    Paper [N]:
    1. Full name of the paper: <title>
    2. Author names: <authors>
    3. Name of the Journal/Conference: <venue>
    4. SCOPUS Index and Quartile Status: <check via Scimago — state if not available in context>
    5. Year of publication: <year>
    6. Type of study: <empirical / conceptual / experimental / review>
    7. Methodology: <describe the approach, model, or technique used>
    8. Findings: <key results, accuracy metrics, contributions>
    9. Limitations and Future scope: <what the authors identified as limitations and future work>
    10. DOI: <doi if available, else state 'Not available in context'>

    Extract as many papers as you can find in the context.
    After all papers, write a 2-3 paragraph synthesis comparing the methodologies and identifying research gaps in Marathi NER.`,
  );

  console.log("📄 Building IEEE document...");
  await generateIEEEDoc(introduction, literatureSurvey);

  console.log("\n✅ Done!");
}

main().catch(console.error);
