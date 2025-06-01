package net.jmp.pinecone.langchain.examples.rag;

/*
 * (#)ReturnSourcesRag.java 0.1.0   06/01/2025
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

import dev.langchain4j.memory.chat.MessageWindowChatMemory;

import dev.langchain4j.model.chat.ChatModel;

import dev.langchain4j.model.embedding.EmbeddingModel;

import dev.langchain4j.model.embedding.onnx.bgesmallenv15q.BgeSmallEnV15QuantizedEmbeddingModel;

import dev.langchain4j.model.openai.OpenAiChatModel;

import static dev.langchain4j.model.openai.OpenAiChatModelName.GPT_4_1;

import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;

import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.Result;

import dev.langchain4j.store.embedding.EmbeddingStore;

import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.nio.file.Path;

import java.util.List;

import static net.jmp.pinecone.langchain.examples.rag.Utils.toPath;

import static net.jmp.util.logging.LoggerUtils.*;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/// The return sources RAG class.
///
/// https://github.com/langchain4j/langchain4j-examples/blob/main/rag-examples/src/main/java/_3_advanced/_09_Advanced_RAG_Return_Sources_Example.java
///
/// @version    0.1.0
/// @since      0.1.0
public final class ReturnSourcesRag implements Runnable, Rag {
    /// The logger.
    private final Logger logger = LoggerFactory.getLogger(this.getClass().getName());

    /// The default constructor.
    public ReturnSourcesRag() {
        super();
    }

    /// The run method.
    @Override
    public void run() {
        if (logger.isTraceEnabled()) {
            logger.trace(entry());
        }

        final String openaiApiKey = System.getProperty("app.openaiApiKey");

        this.logger.info("Return Sources Rag");

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

        final EmbeddingModel embeddingModel = new BgeSmallEnV15QuantizedEmbeddingModel();

        final EmbeddingStore<TextSegment> embeddingStore =
                embed(toPath("documents/miles-of-smiles-terms-of-use.txt"), embeddingModel);

        final ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.6)
                .build();

        // Create an OpenAI chat model

        final ChatModel chatModel = OpenAiChatModel.builder()
                .apiKey(openaiApiKey)
                .modelName(GPT_4_1)
                .logRequests(false)
                .build();

        final ResultAssistant assistant = AiServices.builder(ResultAssistant.class)
                .chatModel(chatModel)               // It should use OpenAI LLM
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .contentRetriever(contentRetriever)
                .build();

        final List<String> questions = List.of(
                "Can I use the car to smuggle drugs?",
                "How do I book my reservation?"
        );

        for (final String question : questions) {
            final Result<String> result = assistant.answer(question);

            this.logger.info("Question: {}", question);
            this.logger.info("Answer  : {}", result.content());

            final List<String> sources = result.sources().stream().map(content -> content.toString()).toList();

            sources.forEach(source -> {
                this.logger.info("Source  : {}", source);
            });
        }

        if (this.logger.isTraceEnabled()) {
            this.logger.trace(exit());
        }
    }

    /// The embed method.
    ///
    /// @param documentPath     java.nio.file.Path
    /// @param embeddingModel   dev.langchain4j.model.embedding.EmbeddingModel
    /// @return                 dev.langchain4j.store.embedding.EmbeddingStore
    private EmbeddingStore<TextSegment> embed(final Path documentPath, final EmbeddingModel embeddingModel) {
        if (this.logger.isTraceEnabled()) {
            this.logger.trace(entryWith(documentPath, embeddingModel));
        }

        final DocumentParser documentParser = new TextDocumentParser();
        final Document document = loadDocument(documentPath, documentParser);

        final DocumentSplitter splitter = DocumentSplitters.recursive(300, 0);
        final List<TextSegment> segments = splitter.split(document);

        final List<Embedding> embeddings = embeddingModel.embedAll(segments).content();

        final EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

        embeddingStore.addAll(embeddings, segments);

        if (this.logger.isTraceEnabled()) {
            this.logger.trace(exitWith(embeddingModel));
        }

        return embeddingStore;
    }

    /// The ResultAssistant interface.
    interface ResultAssistant {
        /// The answer method.
        ///
        /// @param query    java.lang.String
        /// @return         java.util.concurrent.CompletableFuture<java.lang.String>
        Result<String> answer(String query);
    }
}
