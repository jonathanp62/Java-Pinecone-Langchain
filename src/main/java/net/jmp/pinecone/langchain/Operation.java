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
    ///
    /// @param  pineconeApiKey  java.lang.String
    abstract void operate(String pineconeApiKey);

    /// Returns the embedding store.
    ///
    /// @param  pineconeApiKey      java.lang.String
    /// @param  embeddingModelName  java.lang.String
    /// @param  indexName           java.lang.String
    /// @param  namespace           java.lang.String
    /// @return                     dev.langchain4j.store.embedding.EmbeddingStore<dev.langchain4j.data.segment.TextSegment>
    protected EmbeddingStore<TextSegment> getEmbeddingStore(final String pineconeApiKey,
                                               final String embeddingModelName,
                                               final String indexName,
                                               final String namespace) {
        if (this.logger.isTraceEnabled()) {
            this.logger.trace(entryWith(pineconeApiKey, embeddingModelName, indexName, namespace));
        }

        this.logger.info("Creating Pinecone embedding store");

        final EmbeddingModel embeddingModel = this.getEmbeddingModel(embeddingModelName);

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
}
