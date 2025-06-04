package net.jmp.pinecone.langchain.examples.rag;

/*
 * (#)SkipRetrievalRag.java 0.1.0   06/01/2025
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

import dev.langchain4j.data.message.AiMessage;

import dev.langchain4j.data.segment.TextSegment;

import dev.langchain4j.memory.chat.MessageWindowChatMemory;

import dev.langchain4j.model.chat.ChatModel;

import dev.langchain4j.model.embedding.EmbeddingModel;

import dev.langchain4j.model.embedding.onnx.bgesmallenv15q.BgeSmallEnV15QuantizedEmbeddingModel;

import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;

import dev.langchain4j.model.openai.OpenAiChatModel;

import static dev.langchain4j.model.openai.OpenAiChatModelName.GPT_4_1;

import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;

import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;

import dev.langchain4j.rag.query.Query;

import dev.langchain4j.rag.query.router.QueryRouter;

import dev.langchain4j.service.AiServices;

import dev.langchain4j.store.embedding.EmbeddingStore;

import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.nio.file.Path;

import java.util.Collection;
import java.util.List;

import static java.util.Collections.emptyList;
import static java.util.Collections.singletonList;

import static net.jmp.pinecone.langchain.examples.rag.Utils.toPath;

import static net.jmp.util.logging.LoggerUtils.*;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/// The skip retrieval RAG class.
///
/// https://github.com/langchain4j/langchain4j-examples/blob/main/rag-examples/src/main/java/_3_advanced/_06_Advanced_RAG_Skip_Retrieval_Example.java
///
/// @version    0.1.0
/// @since      0.1.0
public final class SkipRetrievalRag implements Runnable, Rag {
    /// The logger.
    private final Logger logger = LoggerFactory.getLogger(this.getClass().getName());

    /// The default constructor.
    public SkipRetrievalRag() {
        super();
    }

    /// The run method.
    @Override
    public void run() {
        if (logger.isTraceEnabled()) {
            logger.trace(entry());
        }

        final String openaiApiKey = System.getProperty("app.openaiApiKey");

        this.logger.info("Skip Retrieval Rag");

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
                .temperature(0.4)
                .logRequests(true)
                .build();

        // Create a query router

        final QueryRouter queryRouter = new QueryRouter() {
            private final PromptTemplate promptTemplate = PromptTemplate.from(
                    "Is the following query related to the business of the car rental company? " +
                            "Answer only 'yes', 'no' or 'maybe'. " +
                            "Query: {{it}}"
            );

            /// The route method.
            ///
            /// @param query dev.langchain4j.rag.query.Query
            /// @return      java.util.Collection<dev.langchain4j.rag.content.retriever.ContentRetriever>
            @Override
            public Collection<ContentRetriever> route(final Query query) {
                final Prompt prompt = this.promptTemplate.apply(query.text());
                final AiMessage aiMessage = chatModel.chat(prompt.toUserMessage()).aiMessage();

                logger.info("LLM decided: {}", aiMessage.text());

                if (aiMessage.text().toLowerCase().contains("no")) {
                    return emptyList();
                }

                return singletonList(contentRetriever);
            }
        };

        final RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        final Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(chatModel)               // It should use OpenAI LLM
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        final List<String> questions = List.of(
                "The dog is black",
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
