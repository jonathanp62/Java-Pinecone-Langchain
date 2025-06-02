package net.jmp.pinecone.langchain;

/*
 * (#)Rag.java  0.1.0   05/29/2025
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

import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.data.segment.TextSegment;

import dev.langchain4j.memory.ChatMemory;

import dev.langchain4j.memory.chat.MessageWindowChatMemory;

import dev.langchain4j.model.chat.ChatModel;

import dev.langchain4j.model.cohere.CohereScoringModel;

import dev.langchain4j.model.embedding.EmbeddingModel;

import dev.langchain4j.model.openai.OpenAiChatModel;

import static dev.langchain4j.model.openai.OpenAiChatModelName.GPT_4_1;

import dev.langchain4j.model.scoring.ScoringModel;

import dev.langchain4j.rag.AugmentationRequest;
import dev.langchain4j.rag.AugmentationResult;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;

import dev.langchain4j.rag.content.aggregator.ContentAggregator;
import dev.langchain4j.rag.content.aggregator.ReRankingContentAggregator;

import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;

import dev.langchain4j.rag.query.Metadata;

import dev.langchain4j.service.AiServices;

import dev.langchain4j.store.embedding.EmbeddingStore;

import static net.jmp.util.logging.LoggerUtils.*;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/// The retrieval-augmented generation class.
///
/// @version    0.1.0
/// @since      0.1.0
final class Rag extends Operation {
    /// The logger.
    private final Logger logger = LoggerFactory.getLogger(this.getClass().getName());

    /// The default constructor.
    Rag() {
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

        this.logger.info("Retrieving/Augmenting/Generating from Pinecone Index: {}", indexName);

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

        final ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(10)
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

        final UserMessage userMessage = new UserMessage(queryText);
        final AugmentationRequest augmentationRequest = new AugmentationRequest(userMessage, Metadata.from(userMessage, null, null));
        final AugmentationResult augmentationResult = retrievalAugmentor.augment(augmentationRequest);

        if (this.logger.isDebugEnabled()) {
            this.logger.debug("Augmented chat message: {}", augmentationResult.chatMessage());
            this.logger.debug("Augmented contents    : {}", augmentationResult.contents());
        }

        final ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(10);

        final ChatModel model = OpenAiChatModel.builder()
                .apiKey(this.getApiKey(openaiApiKey).orElseThrow(() -> new IllegalStateException("Unable to get OpenAI API key")))
                .modelName(GPT_4_1)
                .logRequests(true)
                .build();

        final Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(chatMemory)
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        if (this.logger.isInfoEnabled()) {
            this.logger.info(assistant.chat(queryText));
        }

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
