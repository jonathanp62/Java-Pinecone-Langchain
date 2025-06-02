package net.jmp.pinecone.langchain;

/*
 * (#)Operation.java    0.1.0   05/29/2025
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

import dev.langchain4j.model.embedding.EmbeddingModel;

import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;

import dev.langchain4j.store.embedding.EmbeddingStore;

import dev.langchain4j.store.embedding.pinecone.PineconeEmbeddingStore;
import dev.langchain4j.store.embedding.pinecone.PineconeServerlessIndexConfig;

import java.io.IOException;

import java.nio.file.Files;
import java.nio.file.Paths;

import java.util.Optional;

import static net.jmp.util.logging.LoggerUtils.*;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/// The abstract operation class.
///
/// @version    0.1.0
/// @since      0.1.0
abstract class Operation {
    /// The logger.
    private final Logger logger = LoggerFactory.getLogger(this.getClass().getName());

    /// The default constructor.
    protected Operation() {
        super();
    }

    /// The operate method.
    abstract void operate();

    /// Returns the embedding store.
    ///
    /// @param  pineconeApiKey      java.lang.String
    /// @param  embeddingModel      dev.langchain4j.model.embedding.EmbeddingModel
    /// @param  indexName           java.lang.String
    /// @param  namespace           java.lang.String
    /// @return                     dev.langchain4j.store.embedding.EmbeddingStore<dev.langchain4j.data.segment.TextSegment>
    protected EmbeddingStore<TextSegment> getEmbeddingStore(final String pineconeApiKey,
                                               final EmbeddingModel embeddingModel,
                                               final String indexName,
                                               final String namespace) {
        if (this.logger.isTraceEnabled()) {
            this.logger.trace(entryWith(pineconeApiKey, embeddingModel, indexName, namespace));
        }

        this.logger.info("Creating Pinecone embedding store");

        final EmbeddingStore<TextSegment> embeddingStore = PineconeEmbeddingStore.builder()
                .apiKey(pineconeApiKey)
                .index(indexName)
                .nameSpace(namespace)
                // The index is created if it doesn't exist
                .createIndex(PineconeServerlessIndexConfig.builder()
                        .cloud("AWS")
                        .region("us-east-1")
                        .dimension(embeddingModel.dimension())
                        .build())
                .build();

        if (this.logger.isTraceEnabled()) {
            this.logger.trace(exitWith(embeddingStore));
        }

        return embeddingStore;
    }

    /// Returns the embedding model.
    ///
    /// @param  embeddingModelName  java.lang.String
    /// @return                     dev.langchain4j.model.embedding.EmbeddingModel
    protected EmbeddingModel getEmbeddingModel(final String embeddingModelName) {
        if (this.logger.isTraceEnabled()) {
            this.logger.trace(entryWith(embeddingModelName));
        }

        if (!"all-MiniLM-L6-v2".equals(embeddingModelName)) {
            throw new IllegalArgumentException("Unsupported embedding model: " + embeddingModelName);
        }

        final EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        if (this.logger.isTraceEnabled()) {
            this.logger.trace(exitWith(embeddingModel));
        }

        return embeddingModel;
    }

    /// Get the API key.
    ///
    /// @param  fileName    java.lang.String
    /// @return             java.util.Optional<java.lang.String>
    protected Optional<String> getApiKey(final String fileName) {
        if (this.logger.isTraceEnabled()) {
            this.logger.trace(entryWith(fileName));
        }

        String apiKey = null;

        try {
            apiKey = Files.readString(Paths.get(fileName)).trim();

            if (this.logger.isDebugEnabled()) {
                this.logger.debug("API key file: {}", fileName);
                this.logger.debug("API key: {}", apiKey);
            }
        } catch (final IOException ioe) {
            this.logger.error("Unable to read API key file: {}", fileName, ioe);
        }

        if (this.logger.isTraceEnabled()) {
            this.logger.trace(exitWith(apiKey));
        }

        return Optional.ofNullable(apiKey);
    }
}
