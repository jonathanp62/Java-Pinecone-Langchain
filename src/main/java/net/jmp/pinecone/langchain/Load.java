package net.jmp.pinecone.langchain;

/*
 * (#)Load.java 0.1.0   05/29/2025
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

import com.mongodb.client.*;

import com.mongodb.client.model.Projections;

import dev.langchain4j.data.document.Metadata;

import dev.langchain4j.data.embedding.Embedding;

import dev.langchain4j.data.segment.TextSegment;

import dev.langchain4j.model.embedding.EmbeddingModel;

import dev.langchain4j.store.embedding.EmbeddingStore;

import java.io.IOException;

import java.nio.file.Files;
import java.nio.file.Paths;

import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import static net.jmp.util.logging.LoggerUtils.*;

import org.bson.Document;
import org.bson.conversions.Bson;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/// The load class.
///
/// @version    0.1.0
/// @since      0.1.0
final class Load extends Operation {
    /// The logger.
    private final Logger logger = LoggerFactory.getLogger(this.getClass().getName());

    /// The default constructor.
    Load() {
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
        final String mongoDbCollection = System.getProperty("app.mongoDbCollection");
        final String mongoDbName = System.getProperty("app.mongoDbName");
        final String mongoDbUri = System.getProperty("app.mongoDbUri");
        final String namespace = System.getProperty("app.namespace");
        final String pineconeApiKey = System.getProperty("app.pineconeApiKey");

        this.logger.info("Loading Pinecone Index: {}", indexName);

        if (this.logger.isDebugEnabled()) {
            this.logger.debug("Embedding Model   : {}", embeddingModelName);
            this.logger.debug("Index Name        : {}", indexName);
            this.logger.debug("MongoDB Collection: {}", mongoDbCollection);
            this.logger.debug("MongoDB Name      : {}", mongoDbName);
            this.logger.debug("MongoDB URI       : {}", mongoDbUri);
            this.logger.debug("Namespace         : {}", namespace);
            this.logger.debug("Pinecone Api Key  : {}", pineconeApiKey);}

        final EmbeddingModel embeddingModel = this.getEmbeddingModel(embeddingModelName);

        final EmbeddingStore<TextSegment> embeddingStore = this.getEmbeddingStore(
                this.getApiKey(pineconeApiKey).orElseThrow(() -> new IllegalStateException("Pinecone API key not found")),
                embeddingModel,
                indexName,
                namespace
        );

        final List<TextDocument> textDocuments = this.createContent(mongoDbUri, mongoDbName, mongoDbCollection);

        for (final TextDocument textDocument : textDocuments) {
            final Metadata metadata = Metadata.from(Map.of(
                    "mongoid", textDocument.mongoId,
                    "documentid", textDocument.documentId,
                    "category", textDocument.category
            ));

            final TextSegment textSegment = TextSegment.from(textDocument.content, metadata);
            final Embedding embedding = embeddingModel.embed(textSegment).content();

            embeddingStore.add(embedding, textSegment);
        }

        this.logger.info("Added {} embeddings", textDocuments.size());

        if (this.logger.isTraceEnabled()) {
            this.logger.trace(exit());
        }
    }

    /// Create content from the database.
    ///
    /// @param  dbUri           java.lang.String
    /// @param  dbName          java.lang.String
    /// @param  collectionName  java.lang.String
    /// @return                 java.util.List<net.jmp.pinecone.langchain.Load.TextDocument>
    private List<TextDocument> createContent(final String dbUri,
                                       final String dbName,
                                       final String collectionName) {
        if (this.logger.isTraceEnabled()) {
            this.logger.trace(entryWith(dbUri, dbName, collectionName));
        }

        final List<TextDocument> textDocuments = new LinkedList<>();

        /* Get the content from the database */

        final String uri = this.getMongoDbUri(dbUri).orElseThrow(() -> new IllegalStateException("MongoDB URI not found"));

        try (final MongoClient mongoClient = MongoClients.create(uri)) {
            final MongoDatabase database = mongoClient.getDatabase(dbName);
            final MongoCollection<Document> collection = database.getCollection(collectionName);

            final Bson projectionFields = Projections.fields(
                    Projections.include("id", "content", "category")
            );

            try (final MongoCursor<Document> cursor = collection
                    .find()
                    .projection(projectionFields)
                    .iterator()) {
                if (this.logger.isDebugEnabled()) {
                    this.logger.debug("There are {} documents available", cursor.available());
                }

                while (cursor.hasNext()) {
                    final Document document = cursor.next();
                    final TextDocument textDocument = new TextDocument();

                    textDocument.mongoId = document.get("_id").toString();
                    textDocument.documentId = document.get("id").toString();
                    textDocument.content = document.get("content").toString();
                    textDocument.category = document.get("category").toString();

                    if (this.logger.isDebugEnabled()) {
                        this.logger.debug("MongoId   : {}", textDocument.mongoId);
                        this.logger.debug("DocumentId: {}", textDocument.documentId);
                        this.logger.debug("Content   : {}", textDocument.content);
                        this.logger.debug("Category  : {}", textDocument.category);
                    }

                    textDocuments.add(textDocument);
                }
            }
        }

        if (this.logger.isTraceEnabled()) {
            this.logger.trace(exitWith(textDocuments));
        }

        return textDocuments;
    }

    /// Get the MongoDB URI.
    ///
    /// @param  mongoDbUriFile  java.lang.String
    /// @return                 java.util.Optional<java.lang.String>
    private Optional<String> getMongoDbUri(final String mongoDbUriFile) {
        if (this.logger.isTraceEnabled()) {
            this.logger.trace(entryWith(mongoDbUriFile));
        }

        String mongoDbUri = null;

        try {
            mongoDbUri = Files.readString(Paths.get(mongoDbUriFile)).trim();

            if (this.logger.isDebugEnabled()) {
                this.logger.debug("MongoDb URI file: {}", mongoDbUriFile);
                this.logger.debug("MongoDb URI: {}", mongoDbUri);
            }
        } catch (final IOException ioe) {
            this.logger.error("Unable to read MongoDb URI file: {}", mongoDbUriFile, ioe);
        }

        if (this.logger.isTraceEnabled()) {
            this.logger.trace(exitWith(mongoDbUri));
        }

        return Optional.ofNullable(mongoDbUri);
    }

    /// The text document class.
    private static class TextDocument {
        /// The MongoDb ID.
        private String mongoId;

        /// The document ID.
        private String documentId;

        /// The content.
        private String content;

        /// The category.
        private String category;

        /// The default constructor.
        private TextDocument() {
            super();
        }
    }
}
