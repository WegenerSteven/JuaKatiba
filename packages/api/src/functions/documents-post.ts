import fs from 'node:fs/promises';
import { type HttpRequest, type HttpResponseInit, type InvocationContext, app } from '@azure/functions';
import { AzureOpenAIEmbeddings } from '@langchain/openai';
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { AzureCosmosDBNoSQLVectorStore } from '@langchain/azure-cosmosdb';
import { OllamaEmbeddings } from '@langchain/ollama';
import { FaissStore } from '@langchain/community/vectorstores/faiss';
import 'dotenv/config';
import { BlobServiceClient } from '@azure/storage-blob';
import { badRequest, serviceUnavailable, ok } from '../http-response.js';
import { ollamaEmbeddingsModel, faissStoreFolder } from '../constants.js';
import { getAzureOpenAiTokenProvider, getCredentials } from '../security.js';

export async function postDocuments(request: HttpRequest, context: InvocationContext): Promise<HttpResponseInit> {
  const storageUrl = process.env.AZURE_STORAGE_URL;
  const containerName = process.env.AZURE_STORAGE_CONTAINER_NAME;
  const azureOpenAiEndpoint = process.env.AZURE_OPENAI_API_ENDPOINT;

  try {
    context.log('Starting document upload process');
    // Get the uploaded file from the request
    const parsedForm = await request.formData();

    if (!parsedForm.has('file')) {
      context.log('No file found in form data');
      return badRequest('"file" field not found in form data.');
    }

    // Type mismatch between Node.js FormData and Azure Functions FormData
    const file = parsedForm.get('file') as any as File;
    const filename = file.name;
    context.log(`Processing file: ${filename}, size: ${file.size}`);

    // Extract text from the PDF
    context.log('Starting PDF text extraction');
    const loader = new PDFLoader(file, {
      splitPages: false,
    });
    const rawDocument = await loader.load();
    rawDocument[0].metadata.source = filename;
    context.log(`Extracted ${rawDocument[0].pageContent.length} characters from PDF`);

    // Split the text into smaller chunks
    context.log('Starting text splitting');
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1500,
      chunkOverlap: 100,
    });
    const documents = await splitter.splitDocuments(rawDocument);
    context.log(`Split into ${documents.length} chunks`);

    // Generate embeddings and save in database
    context.log(`Azure OpenAI endpoint: ${azureOpenAiEndpoint ? 'configured' : 'not configured'}`);
    if (azureOpenAiEndpoint) {
      context.log('Using Azure OpenAI for embeddings');
      const credentials = getCredentials();
      const azureADTokenProvider = getAzureOpenAiTokenProvider();

      // Initialize embeddings model and vector database
      const embeddings = new AzureOpenAIEmbeddings({ azureADTokenProvider });
      context.log('Storing documents in Azure Cosmos DB');
      await AzureCosmosDBNoSQLVectorStore.fromDocuments(documents, embeddings, { credentials });
    } else {
      // If no environment variables are set, it means we are running locally
      context.log('No Azure OpenAI endpoint set, using Ollama models and local DB');
      const embeddings = new OllamaEmbeddings({ model: ollamaEmbeddingsModel });
      const folderExists = await checkFolderExists(faissStoreFolder);
      context.log(`Faiss folder exists: ${folderExists}`);
      if (folderExists) {
        context.log('Loading existing Faiss store');
        const store = await FaissStore.load(faissStoreFolder, embeddings);
        await store.addDocuments(documents);
        await store.save(faissStoreFolder);
      } else {
        context.log('Creating new Faiss store');
        const store = await FaissStore.fromDocuments(documents, embeddings, {});
        await store.save(faissStoreFolder);
      }
    }

    context.log(
      `Storage URL: ${storageUrl ? 'configured' : 'not configured'}, Container: ${containerName || 'not configured'}`,
    );
    if (storageUrl && containerName) {
      // Upload the PDF file to Azure Blob Storage
      context.log(`Uploading file to blob storage: "${containerName}/${filename}"`);
      const credentials = getCredentials();
      const blobServiceClient = new BlobServiceClient(storageUrl, credentials);
      const containerClient = blobServiceClient.getContainerClient(containerName);
      const blockBlobClient = containerClient.getBlockBlobClient(filename);
      const buffer = await file.arrayBuffer();
      await blockBlobClient.upload(buffer, file.size, {
        blobHTTPHeaders: { blobContentType: 'application/pdf' },
      });
      context.log('File uploaded to blob storage successfully');
    } else {
      context.log('No Azure Blob Storage connection string set, skipping upload.');
    }

    return ok({ message: 'PDF file uploaded successfully.' });
  } catch (_error: unknown) {
    const error = _error as Error;
    context.error(`Error when processing document-post request: ${error.message}`);
    context.error(`Error stack: ${error.stack}`);

    return serviceUnavailable(`Service temporarily unavailable. Error: ${error.message}`);
  }
}

async function checkFolderExists(folderPath: string): Promise<boolean> {
  try {
    const stats = await fs.stat(folderPath);
    return stats.isDirectory();
  } catch {
    return false;
  }
}

app.http('documents-post', {
  route: 'documents',
  methods: ['POST'],
  authLevel: 'anonymous',
  handler: postDocuments,
});
