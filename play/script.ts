import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf'
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters'
import { OllamaEmbeddings, ChatOllama } from '@langchain/ollama'
import { MemoryVectorStore } from '@langchain/classic/vectorstores/memory'
import { ChatPromptTemplate } from '@langchain/core/prompts'
import { StringOutputParser } from '@langchain/core/output_parsers'
import {
  RunnablePassthrough,
  RunnableSequence,
} from '@langchain/core/runnables'
import { Document as LangDoc } from '@langchain/core/documents'
import * as fs from 'fs'
import * as path from 'path'
import * as unzipper from 'unzipper'

// ─── CONFIG ────────────────────────────────────────────────────────────────
const CONFIG = {
  zipFile: 'ner_papers.zip',
  extractDir: 'extracted',
  outputFile: './ieee-output.docx',
  topic: 'Named Entity Recognition in Marathi Language',
  llmModel: 'mistral:7b',
  embeddingModel: 'nomic-embed-text-v2-moe:latest',
  chunkSize: 1000,
  chunkOverlap: 200,
  retrieverK: 8,
  contextWindow: 16384,
} as const
// ───────────────────────────────────────────────────────────────────────────

// ─── STEP 1: Extract PDFs from ZIP ─────────────────────────────────────────
async function extractPDFsFromZip(
  zipPath: string,
  outputDir: string,
): Promise<string[]> {
  const extractedPaths: string[] = []

  // Ensure output directory exists
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true })
  }

  const directory = await unzipper.Open.file(zipPath)

  for (const entry of directory.files) {
    if (entry.path.endsWith('.pdf')) {
      const fullPath = path.join(outputDir, entry.path)

      // Ensure subdirectories exist if the ZIP has a folder structure
      fs.mkdirSync(path.dirname(fullPath), { recursive: true })

      // Write the file
      const content = await entry.buffer()
      await fs.promises.writeFile(fullPath, content)

      extractedPaths.push(fullPath)
    }
  }

  console.log(`📦 Extracted ${extractedPaths.length} PDF(s) from ZIP\n`)
  return extractedPaths
}

// ─── STEP 2: Load PDFs into LangChain Documents ────────────────────────────
async function loadPDFs(filePaths: string[]): Promise<LangDoc[]> {
  const allDocs: LangDoc[] = []

  for (const filePath of filePaths) {
    if (!fs.existsSync(filePath)) {
      console.warn(`⚠️  Skipping "${filePath}" — file not found`)
      continue
    }

    console.log(`📄 Loading: ${filePath}`)
    const docs = await new PDFLoader(filePath).load()

    // Tag each page with its source filename for traceability
    const taggedDocs = docs.map((doc) => ({
      ...doc,
      metadata: { ...doc.metadata, sourceFile: path.basename(filePath) },
    }))

    allDocs.push(...taggedDocs)
  }

  console.log(
    `✅ Loaded ${allDocs.length} page(s) from ${filePaths.length} file(s)\n`,
  )
  return allDocs
}

// ─── STEP 3: Build the RAG Chain ───────────────────────────────────────────
// RAG = Retrieval-Augmented Generation
// It chunks your docs → embeds them → stores in vector DB → retrieves relevant chunks → feeds to LLM
async function buildRAGChain(docs: LangDoc[]) {
  // Split large docs into smaller chunks so the LLM can digest them
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: CONFIG.chunkSize,
    chunkOverlap: CONFIG.chunkOverlap,
  })
  const chunks = await splitter.splitDocuments(docs)
  console.log(`✂️  Split into ${chunks.length} chunks\n`)

  // Convert chunks to vectors and store in-memory
  const vectorStore = await MemoryVectorStore.fromDocuments(
    chunks,
    new OllamaEmbeddings({ model: CONFIG.embeddingModel }),
  )

  // Retriever fetches the top-K most relevant chunks per query
  const retriever = vectorStore.asRetriever({ k: CONFIG.retrieverK })

  const llm = new ChatOllama({
    model: CONFIG.llmModel,
    temperature: 0.1, // Low temp = focused, factual outputs
    numCtx: CONFIG.contextWindow,
  })

  const prompt = ChatPromptTemplate.fromTemplate(`
You are a senior research assistant specializing in Natural Language Processing,
specifically Named Entity Recognition (NER) in Marathi and other Indian languages.
You are writing an IEEE-format academic paper on the topic: "${CONFIG.topic}".

Use ONLY the provided context from research papers to write your response.
Be precise, factual, and formal in academic tone.
Do not hallucinate or add information not present in the context.

Context:
{context}

Task: {question}
  `)

  // Chain: retrieve relevant chunks → inject into prompt → call LLM → parse output
  const chain = RunnableSequence.from([
    {
      context: retriever.pipe((docs) =>
        docs.map((d) => d.pageContent).join('\n\n'),
      ),
      question: new RunnablePassthrough(),
    },
    prompt,
    llm,
    new StringOutputParser(),
  ])

  return chain
}

// ─── STEP 4: Generate Content via LLM ─────────────────────────────────────
const INTRO_PROMPT = `
Write a detailed IEEE academic introduction section for a paper on "${CONFIG.topic}".
Cover the following points using information from the context:
- What is Named Entity Recognition (NER) and its importance in NLP
- Why Marathi language is significant and challenging for NER
- Challenges specific to Marathi: morphological richness, lack of resources, code-mixing
- Motivation and objectives of this survey
- Brief overview of what the paper covers
Write in formal IEEE academic style. Do not use bullet points — use flowing paragraphs.
Ensure the introduction section is between 1,500 and 3,000 characters long.
`

const LIT_SURVEY_PROMPT = `
Write a detailed IEEE literature survey section for a paper on "${CONFIG.topic}".
For EACH paper found in the context, extract and present the following in this EXACT structured format:

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
After all papers, write a 2-3 paragraph synthesis comparing the methodologies and identifying research gaps in Marathi NER.
`

async function generateSections(
  chain: Awaited<ReturnType<typeof buildRAGChain>>,
  docs: LangDoc[],
) {
  let introduction = ''
  let literatureSurvey = ''

  for (const doc of docs) {
    console.log(`🤖 Generating content for: ${doc.metadata.sourceFile}`)

    const [intro, litSurvey] = await Promise.all([
      chain.invoke(INTRO_PROMPT),
      chain.invoke(LIT_SURVEY_PROMPT),
    ])

    introduction += intro + '\n\n'
    literatureSurvey += litSurvey + '\n\n'
  }

  return { introduction, literatureSurvey }
}

// ─── STEP 5: Build IEEE Word Document ─────────────────────────────────────
async function buildIEEEDocument(
  introduction: string,
  literatureSurvey: string,
) {
  const { Document, Packer, Paragraph, TextRun, AlignmentType, HeadingLevel } =
    await import('docx')

  // Helper: turn a multi-line string into an array of justified paragraphs
  const toBodyParagraphs = (text: string) =>
    text
      .split('\n')
      .filter((line) => line.trim().length > 0)
      .map(
        (line) =>
          new Paragraph({
            alignment: AlignmentType.BOTH,
            spacing: { before: 0, after: 120 },
            indent: { firstLine: 720 },
            children: [
              new TextRun({
                text: line.trim(),
                size: 20,
                font: 'Times New Roman',
              }),
            ],
          }),
      )

  // Helper: create a numbered section heading (e.g. "I. Introduction")
  const sectionHeading = (title: string) =>
    new Paragraph({
      heading: HeadingLevel.HEADING_1,
      spacing: { before: 240, after: 120 },
      children: [
        new TextRun({
          text: title,
          bold: true,
          size: 22,
          font: 'Times New Roman',
        }),
      ],
    })

  const sources = introduction.split('\n').filter((p) => p.trim().length > 0)

  const doc = new Document({
    styles: {
      default: {
        document: { run: { font: 'Times New Roman', size: 20 } },
      },
      paragraphStyles: [
        {
          id: 'Heading1',
          name: 'Heading 1',
          basedOn: 'Normal',
          next: 'Normal',
          run: { size: 22, bold: true, font: 'Times New Roman' },
          paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 0 },
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
          // Title
          new Paragraph({
            alignment: AlignmentType.CENTER,
            spacing: { before: 0, after: 240 },
            children: [
              new TextRun({
                text: 'Named Entity Recognition in Marathi Language: A Systematic Literature Survey',
                bold: true,
                size: 28,
                font: 'Times New Roman',
              }),
            ],
          }),

          // Subtitle / Author line
          new Paragraph({
            alignment: AlignmentType.CENTER,
            spacing: { before: 0, after: 480 },
            children: [
              new TextRun({
                text: 'Generated via RAG Pipeline · LangChain + Ollama',
                italics: true,
                size: 20,
                font: 'Times New Roman',
              }),
            ],
          }),

          // Abstract heading
          new Paragraph({
            alignment: AlignmentType.CENTER,
            spacing: { before: 0, after: 120 },
            children: [
              new TextRun({
                text: 'Abstract',
                bold: true,
                italics: true,
                size: 20,
                font: 'Times New Roman',
              }),
            ],
          }),

          // Abstract body
          new Paragraph({
            alignment: AlignmentType.BOTH,
            spacing: { before: 0, after: 480 },
            children: [
              new TextRun({
                text: `This paper presents a systematic literature survey on Named Entity Recognition (NER) in the Marathi language. NER is a fundamental task in Natural Language Processing (NLP) that identifies and classifies named entities such as persons, locations, and organizations in text. Marathi, a morphologically rich Indic language spoken by over 83 million people, poses unique challenges for NER due to its complex morphology, limited annotated corpora, and code-mixing phenomena. This survey synthesizes findings from ${sources.length} research paper(s) to provide a comprehensive overview of existing methodologies, datasets, and future directions in Marathi NER.`,
                size: 20,
                font: 'Times New Roman',
              }),
            ],
          }),

          // Section I: Introduction
          sectionHeading('I. Introduction'),
          ...toBodyParagraphs(introduction),

          // Section II: Literature Survey
          sectionHeading('II. Literature Survey'),
          ...toBodyParagraphs(literatureSurvey),

          // References
          sectionHeading('References'),
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
                    font: 'Times New Roman',
                  }),
                  new TextRun({
                    text: `Source document: "${src}". Processed via RAG pipeline, ${new Date().getFullYear()}.`,
                    size: 20,
                    font: 'Times New Roman',
                  }),
                ],
              }),
          ),
        ],
      },
    ],
  })

  const buffer = await Packer.toBuffer(doc)
  fs.writeFileSync(CONFIG.outputFile, buffer)
  console.log(`\n📝 IEEE document saved to: ${CONFIG.outputFile}`)
}

// ─── MAIN ──────────────────────────────────────────────────────────────────
async function main() {
  console.log('🚀 Starting RAG pipeline...\n')

  const pdfPaths = await extractPDFsFromZip(CONFIG.zipFile, CONFIG.extractDir)

  if (pdfPaths.length === 0) {
    console.error('❌ No PDF files found in the ZIP file.')
    process.exit(1)
  }

  const docs = await loadPDFs(pdfPaths)

  if (docs.length === 0) {
    console.error('❌ No documents loaded. Check your extracted PDF paths.')
    process.exit(1)
  }

  const chain = await buildRAGChain(docs)

  console.log('📄 Generating sections...\n')
  const { introduction, literatureSurvey } = await generateSections(chain, docs)

  console.log('📄 Building IEEE document...')
  await buildIEEEDocument(introduction, literatureSurvey)

  console.log('\n✅ Done!')
}

main().catch(console.error)
