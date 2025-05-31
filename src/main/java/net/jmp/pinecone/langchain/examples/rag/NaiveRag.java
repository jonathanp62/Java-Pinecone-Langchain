package net.jmp.pinecone.langchain.examples.rag;

/*
 * (#)NaiveRag.java 0.1.0   05/30/2025
 *
 * @author   Jonathan Parker
 *
 * MIT License
 *
 * Copyright (c) 2025 Jonathan M. Parker
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;

import static dev.langchain4j.data.document.loader.FileSystemDocumentLoader.loadDocument;

import dev.langchain4j.data.document.parser.TextDocumentParser;

import dev.langchain4j.data.document.splitter.DocumentSplitters;

import dev.langchain4j.data.embedding.Embedding;

import dev.langchain4j.data.segment.TextSegment;

import dev.langchain4j.memory.ChatMemory;

import dev.langchain4j.memory.chat.MessageWindowChatMemory;

import dev.langchain4j.model.chat.ChatModel;

import dev.langchain4j.model.embedding.EmbeddingModel;

import dev.langchain4j.model.embedding.onnx.bgesmallenv15q.BgeSmallEnV15QuantizedEmbeddingModel;

import dev.langchain4j.model.openai.OpenAiChatModel;

import static dev.langchain4j.model.openai.OpenAiChatModelName.GPT_4_1;

import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;

import dev.langchain4j.service.AiServices;

import dev.langchain4j.store.embedding.EmbeddingStore;

import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.List;

import static net.jmp.pinecone.langchain.examples.rag.Utils.toPath;

import static net.jmp.util.logging.LoggerUtils.*;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/// The naive RAG class.
///
/// https://github.com/langchain4j/langchain4j-examples/blob/main/rag-examples/src/main/java/_2_naive/Naive_RAG_Example.java
///
/// @version    0.1.0
/// @since      0.1.0
public final class NaiveRag implements Runnable, Rag {
    /// The logger.
    private final Logger logger = LoggerFactory.getLogger(this.getClass().getName());

    /// The default constructor.
    public NaiveRag() {
        super();
    }

    /// The run method.
    @Override
    public void run() {
        if (this.logger.isTraceEnabled()) {
            this.logger.trace(entry());
        }

        final String openaiApiKey = System.getProperty("app.openaiApiKey");

        this.logger.info("Naive Rag");

        if (this.logger.isDebugEnabled()) {
            this.logger.debug("OpenAI Api Key  : {}", openaiApiKey);
        }

        this.rag(Utils.getApiKey(openaiApiKey));

        if (this.logger.isTraceEnabled()) {
            this.logger.trace(exit());
        }
    }

    /// The rag method.
    ///
    /// @param openaiApiKey java.lang.String
    @Override
    public void rag(final String openaiApiKey) {
        if (this.logger.isTraceEnabled()) {
            this.logger.trace(entryWith(openaiApiKey));
        }

        final String documentPath = "documents/miles-of-smiles-terms-of-use.txt";
        final DocumentParser documentParser = new TextDocumentParser();
        final Document document = loadDocument(toPath(documentPath), documentParser);

        /*
         * Now, we need to split this document into smaller segments, also known as "chunks."
         * This approach allows us to send only relevant segments to the LLM in response to a user query,
         * rather than the entire document. For instance, if a user asks about cancellation policies,
         * we will identify and send only those segments related to cancellation.
         * A good starting point is to use a recursive document splitter that initially attempts
         * to split by paragraphs. If a paragraph is too large to fit into a single segment,
         * the splitter will recursively divide it by newlines, then by sentences, and finally by words,
         * if necessary, to ensure each piece of text fits into a single segment.
         */

        final DocumentSplitter splitter = DocumentSplitters.recursive(300, 0);
        final List<TextSegment> segments = splitter.split(document);

        if (this.logger.isDebugEnabled()) {
            this.logger.debug("Number of segments: {}", segments.size());

            for (final TextSegment segment : segments) {
                this.logger.debug("Segment: {}", segment);
            }
        }

        /*
         * Now, we need to embed (also known as "vectorize") these segments.
         * Embedding is needed for performing similarity searches.
         * For this example, we'll use a local in-process embedding model, but you can choose any supported model.
         * Langchain4j currently supports more than 10 popular embedding model providers.
         */

        final EmbeddingModel embeddingModel = new BgeSmallEnV15QuantizedEmbeddingModel();
        final List<Embedding> embeddings = embeddingModel.embedAll(segments).content();

        if (this.logger.isDebugEnabled()) {
            this.logger.debug("Number of embeddings: {}", embeddings.size());

            for (final Embedding embedding : embeddings) {
                this.logger.debug("Embedding: {}", embedding);
            }
        }

        /*
         * Next, we will store these embeddings in an embedding store (also known as a "vector database").
         * This store will be used to search for relevant segments during each interaction with the LLM.
         * For simplicity, this example uses an in-memory embedding store, but you can choose from any supported store.
         * Langchain4j currently supports more than 15 popular embedding stores.
         */

        final EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

        embeddingStore.addAll(embeddings, segments);

        /*
         * The content retriever is responsible for retrieving relevant content based on a user query.
         * Currently, it is capable of retrieving text segments, but future enhancements will include support for
         * additional modalities like images, audio, and more.
         */

        final ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2) // On each interaction we will retrieve the 2 most relevant segments
                .minScore(0.5) // We want to retrieve segments at least somewhat similar to user query
                .build();

        // Create an OpenAI chat model

        final ChatModel chatModel = OpenAiChatModel.builder()
                .apiKey(openaiApiKey)
                .modelName(GPT_4_1)
                .build();

        /*
         * Optionally, we can use a chat memory, enabling back-and-forth conversation
         * with the LLM and allowing it to remember previous interactions.
         * Currently, LangChain4j offers two chat memory implementations:
         * MessageWindowChatMemory and TokenWindowChatMemory.
         */

        final ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(2);

        // Create an assistant that has access to our documents

        final Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(chatModel)               // It should use OpenAI LLM
                .chatMemory(chatMemory)             // It should remember previous interactions
                .contentRetriever(contentRetriever) // It should have access to our documents
                .build();

        final List<String> questions = List.of(
                "Can I cancel my reservation?",
                "I had an accident, should I pay extra?"
        );

        for (final String question : questions) {
            final String answer = assistant.chat(question);

            this.logger.info("Question: {}", question);
            this.logger.info("Answer  : {}", answer);
        }

        if (this.logger.isTraceEnabled()) {
            this.logger.trace(exit());
        }
    }
}
