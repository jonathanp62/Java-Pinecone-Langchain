package net.jmp.pinecone.langchain.examples.rag;

/*
 * (#)EasyRag.java  0.1.0   05/30/2025
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

import dev.langchain4j.data.segment.TextSegment;

import static dev.langchain4j.data.document.loader.FileSystemDocumentLoader.loadDocuments;

import static dev.langchain4j.model.openai.OpenAiChatModelName.GPT_4_1;

import dev.langchain4j.model.chat.ChatModel;

import dev.langchain4j.model.openai.OpenAiChatModel;

import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;

import dev.langchain4j.service.AiServices;

import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;

import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.List;

import static net.jmp.pinecone.langchain.examples.rag.Utils.glob;
import static net.jmp.pinecone.langchain.examples.rag.Utils.toPath;

import static net.jmp.util.logging.LoggerUtils.*;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/// The easy RAG class.
///
/// @version    0.1.0
/// @since      0.1.0
public final class EasyRag implements Runnable {
    /// The logger.
    private final Logger logger = LoggerFactory.getLogger(this.getClass().getName());

    /// The default constructor.
    public EasyRag() {
        super();
    }

    /// The run method.
    @Override
    public void run() {
        if (this.logger.isTraceEnabled()) {
            this.logger.trace(entry());
        }

        final String openaiApiKey = System.getProperty("app.openaiApiKey");

        this.logger.info("Easy Rag");

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
    private void rag(final String openaiApiKey) {
        if (this.logger.isTraceEnabled()) {
            this.logger.trace(entry());
        }

        final List<Document> documents = loadDocuments(toPath("documents/"), glob("*.txt"));

        // Create an empty in-memory store for our documents and their embeddings

        final InMemoryEmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

        // Ingest our documents into the store

        EmbeddingStoreIngestor.ingest(documents, embeddingStore);

        // Create a content retriever from the embedding store

        final ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.from(embeddingStore);

        // Create an OpenAI chat model

        final ChatModel chatModel = OpenAiChatModel.builder()
                .apiKey(openaiApiKey)
                .modelName(GPT_4_1)
                .build();

        // Create an assistant that has access to our documents

        final Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(chatModel)               // It should use OpenAI LLM
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
