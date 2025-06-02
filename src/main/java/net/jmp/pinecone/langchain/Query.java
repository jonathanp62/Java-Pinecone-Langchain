package net.jmp.pinecone.langchain;

/*
 * (#)Query.java    0.1.0   05/29/2025
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

import dev.langchain4j.data.embedding.Embedding;

import dev.langchain4j.data.segment.TextSegment;

import dev.langchain4j.model.embedding.EmbeddingModel;

import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.EmbeddingStore;

import static net.jmp.util.logging.LoggerUtils.*;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/// The query class.
///
/// @version    0.1.0
/// @since      0.1.0
final class Query extends Operation {
    /// The logger.
    private final Logger logger = LoggerFactory.getLogger(this.getClass().getName());

    /// The default constructor.
    Query() {
        super();
    }

    /// The operate method.
    @Override
    void operate() {
        if (this.logger.isTraceEnabled()) {
            this.logger.trace(entry());
        }

        final String embeddingModelName = System.getProperty("app.embeddingModel");
        final String indexName = System.getProperty("app.indexName");
        final String namespace = System.getProperty("app.namespace");
        final String pineconeApiKey = System.getProperty("app.pineconeApiKey");
        final String queryText = System.getProperty("app.queryText");

        this.logger.info("Querying Pinecone Index: {}", indexName);

        if (this.logger.isDebugEnabled()) {
            this.logger.debug("Embedding Model : {}", embeddingModelName);
            this.logger.debug("Index Name      : {}", indexName);
            this.logger.debug("Namespace       : {}", namespace);
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

        final Embedding queryEmbedding = embeddingModel.embed(queryText).content();

        final EmbeddingSearchRequest searchRequest = EmbeddingSearchRequest.builder()
                .queryEmbedding(queryEmbedding)
                .maxResults(10)
                .build();

        final EmbeddingSearchResult<TextSegment> searchResult = embeddingStore.search(searchRequest);

        for (final EmbeddingMatch<TextSegment> match : searchResult.matches()) {
            this.logger.info("{} (Score: {})", match.embedded().text(), match.score());
        }

        if (this.logger.isTraceEnabled()) {
            this.logger.trace(exit());
        }
    }
}
