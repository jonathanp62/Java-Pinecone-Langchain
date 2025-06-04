package net.jmp.pinecone.langchain;

/*
 * (#)HybridQuery.java  0.1.0   06/04/2025
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

import dev.langchain4j.data.segment.TextSegment;

import dev.langchain4j.memory.ChatMemory;

import dev.langchain4j.memory.chat.MessageWindowChatMemory;

import dev.langchain4j.model.chat.ChatModel;

import dev.langchain4j.model.cohere.CohereScoringModel;

import dev.langchain4j.model.embedding.EmbeddingModel;

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

import dev.langchain4j.store.embedding.filter.Filter;

import static dev.langchain4j.store.embedding.filter.MetadataFilterBuilder.metadataKey;
import static net.jmp.util.logging.LoggerUtils.*;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/// The retrieval-augmented generation class.
///
/// @version    0.1.0
/// @since      0.1.0

final class HybridQuery extends Operation {
    /// The logger.
    private final Logger logger = LoggerFactory.getLogger(this.getClass().getName());

    /// The default constructor.
    HybridQuery() {
        super();
    }

    /// The operate method.
    @Override
    void operate() {
        if (this.logger.isTraceEnabled()) {
            this.logger.trace(entry());
        }

        final String cohereApiKey = System.getProperty("app.cohereApiKey");
        final String embeddingModelName = System.getProperty("app.embeddingModel");
        final String indexName = System.getProperty("app.indexName");
        final String namespace = System.getProperty("app.namespace");
        final String openaiApiKey = System.getProperty("app.openaiApiKey");
        final String pineconeApiKey = System.getProperty("app.pineconeApiKey");
        final String queryText = System.getProperty("app.queryText");

        this.logger.info("Hybrid Query from Pinecone Index: {}", indexName);

        if (this.logger.isDebugEnabled()) {
            this.logger.debug("Cohere Api Key  : {}", cohereApiKey);
            this.logger.debug("Embedding Model : {}", embeddingModelName);
            this.logger.debug("Index Name      : {}", indexName);
            this.logger.debug("Namespace       : {}", namespace);
            this.logger.debug("OpenAI Api Key  : {}", openaiApiKey);
            this.logger.debug("Pinecone Api Key: {}", pineconeApiKey);
            this.logger.debug("Query Text      : {}", queryText);
        }

        final EmbeddingModel embeddingModel = this.getEmbeddingModel(embeddingModelName);

        final EmbeddingStore<TextSegment> embeddingStore = this.getEmbeddingStore(
                this.getApiKey(pineconeApiKey).orElseThrow(() -> new IllegalStateException("Pinecone API key not found")),
                embeddingModel,
                indexName,
                namespace
        );

        /*
         * Not supported by Pinecone:
         *   final Filter containsLight = metadataKey("text_segment").containsString("light");
         */

        final Filter lightCategories = metadataKey("category")
                .isEqualTo("astronomy")
                .or(metadataKey("category").isEqualTo("inventions"))
                .or(metadataKey("category").isEqualTo("physics"))
                .or(metadataKey("category").isEqualTo("science"))
                .or(metadataKey("category").isEqualTo("technology"));

        final ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .filter(lightCategories)   // Limit the search to some categories
                .maxResults(25)
                .build();

        final ScoringModel scoringModel = CohereScoringModel.builder()
                .apiKey(this.getApiKey(cohereApiKey).orElseThrow(() -> new IllegalStateException("Unable to get Cohere API key")))
                .modelName("rerank-english-v3.0")
                .build();

        final ContentAggregator contentAggregator = ReRankingContentAggregator.builder()
                .scoringModel(scoringModel)
                .build();

        final RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .contentRetriever(contentRetriever)
                .contentAggregator(contentAggregator)
                .build();

        final ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(10);

        final ChatModel chatModel = OpenAiChatModel.builder()
                .apiKey(this.getApiKey(openaiApiKey).orElseThrow(() -> new IllegalStateException("Unable to get OpenAI API key")))
                .modelName(GPT_4_1)
                .temperature(0.4)
                .logRequests(true)
                .logResponses(true)
                .build();

        final Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(chatModel)
                .chatMemory(chatMemory)
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        final String answer = assistant.chat(queryText);

        this.logger.info("Question: {}", queryText);
        this.logger.info("Answer  : {}", answer);

        if (this.logger.isTraceEnabled()) {
            this.logger.trace(exit());
        }
    }

    /// The assistant interface.
    interface Assistant {
        /// The chat method.
        ///
        /// @param  message  java.lang.String
        /// @return          java.lang.String
        String chat(String message);
    }
}
