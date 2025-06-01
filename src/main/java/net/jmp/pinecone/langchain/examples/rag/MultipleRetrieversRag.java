package net.jmp.pinecone.langchain.examples.rag;

/*
 * (#)MultipleRetrieversRag.java    0.1.0   06/01/2025
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

import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;

import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;

import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;

import dev.langchain4j.service.AiServices;

import dev.langchain4j.store.embedding.EmbeddingStore;

import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.nio.file.Path;
import java.util.List;

import static net.jmp.pinecone.langchain.examples.rag.Utils.toPath;

import static net.jmp.util.logging.LoggerUtils.*;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/// The multiple retrievers RAG class.
///
/// https://github.com/langchain4j/langchain4j-examples/blob/main/rag-examples/src/main/java/_3_advanced/_07_Advanced_RAG_Multiple_Retrievers_Example.java
///
/// @version    0.1.0
/// @since      0.1.0
public final class MultipleRetrieversRag implements Runnable, Rag {
    /// The logger.
    private final Logger logger = LoggerFactory.getLogger(this.getClass().getName());

    /// The default constructor.
    public MultipleRetrieversRag() {
        super();
    }

    /// The run method.
    @Override
    public void run() {
        if (logger.isTraceEnabled()) {
            logger.trace(entry());
        }

        final String openaiApiKey = System.getProperty("app.openaiApiKey");

        this.logger.info("Multiple Retrievers Rag");

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

        // Create our first content retriever for terms of use

        final EmbeddingStore<TextSegment> embeddingStore1 =
                embed(toPath("documents/miles-of-smiles-terms-of-use.txt"), embeddingModel);

        final ContentRetriever contentRetriever1 = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore1)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.6)
                .build();

        // Create our second content retriever for John Doe's biography

        final EmbeddingStore<TextSegment> embeddingStore2 =
                embed(toPath("documents/biography-of-john-doe.txt"), embeddingModel);

        final ContentRetriever contentRetriever2 = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore2)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.6)
                .build();

        // Create a query router that will route each query to both retrievers

        final QueryRouter queryRouter = new DefaultQueryRouter(contentRetriever1, contentRetriever2);

        final RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        // Create an OpenAI chat model

        final ChatModel chatModel = OpenAiChatModel.builder()
                .apiKey(openaiApiKey)
                .modelName(GPT_4_1)
                .logRequests(true)
                .build();

        final Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(chatModel)               // It should use OpenAI LLM
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        final List<String> questions = List.of(
                "What did John Doe do in his spare time?",
                "How do I book my reservation?"
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
}
