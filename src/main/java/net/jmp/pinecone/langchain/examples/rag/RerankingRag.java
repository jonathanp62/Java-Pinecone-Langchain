package net.jmp.pinecone.langchain.examples.rag;

/*
 * (#)RerankingRag.java 0.1.0   05/31/2025
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

import static dev.langchain4j.data.document.loader.FileSystemDocumentLoader.loadDocument;

import dev.langchain4j.data.document.parser.TextDocumentParser;

import dev.langchain4j.data.document.splitter.DocumentSplitters;

import dev.langchain4j.data.segment.TextSegment;

import dev.langchain4j.memory.ChatMemory;

import dev.langchain4j.memory.chat.MessageWindowChatMemory;

import dev.langchain4j.model.chat.ChatModel;

import dev.langchain4j.model.cohere.CohereScoringModel;

import dev.langchain4j.model.embedding.EmbeddingModel;

import dev.langchain4j.model.embedding.onnx.bgesmallenv15q.BgeSmallEnV15QuantizedEmbeddingModel;

import dev.langchain4j.model.openai.OpenAiChatModel;

import static dev.langchain4j.model.openai.OpenAiChatModelName.GPT_4_1;

import dev.langchain4j.model.scoring.ScoringModel;

import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;

import dev.langchain4j.rag.content.aggregator.ContentAggregator;
import dev.langchain4j.rag.content.aggregator.ReRankingContentAggregator;

import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;

import dev.langchain4j.service.AiServices;

import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;

import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.List;

import static net.jmp.pinecone.langchain.examples.rag.Utils.toPath;

import static net.jmp.util.logging.LoggerUtils.*;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/// The reranking RAG class.
///
/// https://github.com/langchain4j/langchain4j-examples/blob/main/rag-examples/src/main/java/_3_advanced/_03_Advanced_RAG_with_ReRanking_Example.java
///
/// @version    0.1.0
/// @since      0.1.0
public final class RerankingRag implements Runnable, Rag {
    /// The logger.
    private final Logger logger = LoggerFactory.getLogger(this.getClass().getName());

    /// The Cohere API key.
    private String cohereApiKey;

    /// The default constructor.
    public RerankingRag() {
        super();
    }

    /// The run method.
    @Override
    public void run() {
        if (logger.isTraceEnabled()) {
            logger.trace(entry());
        }

        final String cohereApiKeyFile = System.getProperty("app.cohereApiKey");
        final String openaiApiKey = System.getProperty("app.openaiApiKey");

        this.logger.info("Reranking Rag");

        if (this.logger.isDebugEnabled()) {
            this.logger.debug("Cohere Api Key  : {}", cohereApiKeyFile);
            this.logger.debug("OpenAI Api Key  : {}", openaiApiKey);
        }

        this.cohereApiKey = Utils.getApiKey(cohereApiKeyFile);

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
        final Document document = loadDocument(toPath(documentPath), new TextDocumentParser());
        final EmbeddingModel embeddingModel = new BgeSmallEnV15QuantizedEmbeddingModel();
        final EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

        final EmbeddingStoreIngestor ingestor = EmbeddingStoreIngestor.builder()
                .documentSplitter(DocumentSplitters.recursive(300, 0))
                .embeddingModel(embeddingModel)
                .embeddingStore(embeddingStore)
                .build();

        ingestor.ingest(document);

        final ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(5)
                .build();

        // Reranking is also known as scoring

        final ScoringModel scoringModel = CohereScoringModel.builder()
                .apiKey(this.cohereApiKey)
                .modelName("rerank-english-v3.0")
                .build();

        // Make sure only most relevant and non-redundant content is returned

        final ContentAggregator contentAggregator = ReRankingContentAggregator.builder()
                .scoringModel(scoringModel)
                .minScore(0.8)
                .build();

        final RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .contentRetriever(contentRetriever)
                .contentAggregator(contentAggregator)
                .build();

        // Create an OpenAI chat model

        final ChatModel chatModel = OpenAiChatModel.builder()
                .apiKey(openaiApiKey)
                .modelName(GPT_4_1)
                .logRequests(true)
                .build();

        final ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(2);

        final Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(chatModel)               // It should use OpenAI LLM
                .chatMemory(chatMemory)             // It should remember previous interactions
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        final List<String> questions = List.of(
                "Hello!",
                "Can I cancel my reservation?"
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
