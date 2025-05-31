package net.jmp.pinecone.langchain.examples.rag;

/*
 * (#)MetadataFilteringRag.java 0.1.0   05/31/2025
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

import static dev.langchain4j.data.document.Metadata.metadata;

import dev.langchain4j.data.segment.TextSegment;

import dev.langchain4j.memory.chat.ChatMemoryProvider;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;

import dev.langchain4j.model.chat.ChatModel;

import dev.langchain4j.model.embedding.EmbeddingModel;

import dev.langchain4j.model.embedding.onnx.bgesmallenv15q.BgeSmallEnV15QuantizedEmbeddingModel;

import dev.langchain4j.model.openai.OpenAiChatModel;

import static dev.langchain4j.model.openai.OpenAiChatModelName.GPT_4_1;

import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;

import dev.langchain4j.rag.query.Query;

import dev.langchain4j.service.AiServices;

import dev.langchain4j.store.embedding.EmbeddingStore;

import dev.langchain4j.store.embedding.filter.Filter;

import static dev.langchain4j.store.embedding.filter.MetadataFilterBuilder.metadataKey;

import dev.langchain4j.store.embedding.filter.builder.sql.LanguageModelSqlFilterBuilder;
import dev.langchain4j.store.embedding.filter.builder.sql.TableDefinition;

import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import dev.langchain4j.store.memory.chat.InMemoryChatMemoryStore;

import java.util.List;

import java.util.function.Function;

import static net.jmp.util.logging.LoggerUtils.*;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/// The metadata filtering RAG class.
///
/// https://github.com/langchain4j/langchain4j-examples/blob/main/rag-examples/src/main/java/_3_advanced/_05_Advanced_RAG_with_Metadata_Filtering_Examples.java
///
/// @version    0.1.0
/// @since      0.1.0
public final class MetadataFilteringRag implements Runnable, Rag {
    /// The logger.
    private final Logger logger = LoggerFactory.getLogger(this.getClass().getName());

    /// The default constructor.
    public MetadataFilteringRag() {
        super();
    }

    /// The run method.
    @Override
    public void run() {
        if (logger.isTraceEnabled()) {
            logger.trace(entry());
        }

        final String openaiApiKey = System.getProperty("app.openaiApiKey");

        this.logger.info("Metadata Filtering Rag");

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

        // Create an OpenAI chat model

        final ChatModel chatModel = OpenAiChatModel.builder()
                .apiKey(openaiApiKey)
                .modelName(GPT_4_1)
                .logRequests(true)
                .build();

        this.staticFilter(embeddingModel, chatModel);
        this.dynamicFilter(embeddingModel, chatModel);
        this.llmFilter(embeddingModel, chatModel);

        if (this.logger.isTraceEnabled()) {
            this.logger.trace(exit());
        }
    }

    /// Filter statically.
    ///
    /// @param embeddingModel   dev.langchain4j.model.embedding.EmbeddingModel
    /// @param chatModel        dev.langchain4j.model.chat.ChatModel
    private void staticFilter(final EmbeddingModel embeddingModel, final ChatModel chatModel) {
        if (this.logger.isTraceEnabled()) {
            this.logger.trace(entryWith(embeddingModel, chatModel));
        }

        final TextSegment dogsSegment = TextSegment.from("This is an article about dogs.", metadata("animal", "dog"));
        final TextSegment birdsSegment = TextSegment.from("This is an article about birds.", metadata("animal", "bird"));

        final EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

        embeddingStore.add(embeddingModel.embed(dogsSegment).content(), dogsSegment);
        embeddingStore.add(embeddingModel.embed(birdsSegment).content(), birdsSegment);

        final Filter onlyDogs = metadataKey("animal").isEqualTo("dog");

        final ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .filter(onlyDogs)   // Limit the search to segments only about dogs
                .build();

        final Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(chatModel)               // It should use OpenAI LLM
                .contentRetriever(contentRetriever)
                .build();

        final String question = "Which animal?";
        final String answer = assistant.chat(question);

        this.logger.info("Question: {}", question);
        this.logger.info("Answer  : {}", answer);

        if (this.logger.isTraceEnabled()) {
            this.logger.trace(exit());
        }
    }

    /// Filter dynamically.
    ///
    /// @param embeddingModel   dev.langchain4j.model.embedding.EmbeddingModel
    /// @param chatModel        dev.langchain4j.model.chat.ChatModel
    private void dynamicFilter(final EmbeddingModel embeddingModel, final ChatModel chatModel) {
        if (this.logger.isTraceEnabled()) {
            this.logger.trace(entryWith(embeddingModel, chatModel));
        }

        final TextSegment user1Info = TextSegment.from("My favorite color is green", metadata("userId", "1"));
        final TextSegment user2Info = TextSegment.from("My favorite color is red", metadata("userId", "2"));

        final EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

        embeddingStore.add(embeddingModel.embed(user1Info).content(), user1Info);
        embeddingStore.add(embeddingModel.embed(user2Info).content(), user2Info);

        // Define a function that returns a filter based on the user ID

        final Function<Query, Filter> filterByUserId =
                (query) -> metadataKey("userId").isEqualTo(query.metadata().chatMemoryId().toString());

        final ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .dynamicFilter(filterByUserId)   // Limit the search to segments only about the user
                .build();

        final ChatMemoryProvider chatMemoryProvider = memoryId -> MessageWindowChatMemory.builder()
                .id(memoryId)
                .maxMessages(5)
                .chatMemoryStore(new InMemoryChatMemoryStore())
                .build();

        final PersonalizedAssistant assistant = AiServices.builder(PersonalizedAssistant.class)
                .chatModel(chatModel)
                .chatMemoryProvider(chatMemoryProvider)
                .contentRetriever(contentRetriever)
                .build();

        final List<UserQuestion> userQuestions = List.of(
                new UserQuestion("1", "Which color would be best for a dress?"),
                new UserQuestion("2", "Which color would be best for a suit?"));

        for (final UserQuestion userQuestion : userQuestions) {
            this.logger.info("Question: {}", userQuestion.question());
            this.logger.info("Answer  : {}", assistant.chat(userQuestion.userId(), userQuestion.question()));
        }

        if (this.logger.isTraceEnabled()) {
            this.logger.trace(exit());
        }
    }

    /// Filter by LLM generated metadata.
    ///
    /// @param embeddingModel   dev.langchain4j.model.embedding.EmbeddingModel
    /// @param chatModel        dev.langchain4j.model.chat.ChatModel
    private void llmFilter(final EmbeddingModel embeddingModel, final ChatModel chatModel) {
        if (this.logger.isTraceEnabled()) {
            this.logger.trace(entryWith(embeddingModel, chatModel));
        }

        final TextSegment forrestGump = TextSegment.from("Forrest Gump", metadata("genre", "drama").put("year", 1994));
        final TextSegment groundhogDay = TextSegment.from("Groundhog Day", metadata("genre", "comedy").put("year", 1993));
        final TextSegment dieHard = TextSegment.from("Die Hard", metadata("genre", "action").put("year", 1998));

        // Describe metadata keys as if they were columns in the SQL table

        final TableDefinition tableDefinition = TableDefinition.builder()
                .name("movies")
                .addColumn("genre", "VARCHAR", "one of: [comedy, drama, action]")
                .addColumn("year", "INT")
                .build();

        final LanguageModelSqlFilterBuilder sqlFilterBuilder = new LanguageModelSqlFilterBuilder(chatModel, tableDefinition);

        final EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

        embeddingStore.add(embeddingModel.embed(forrestGump).content(), forrestGump);
        embeddingStore.add(embeddingModel.embed(groundhogDay).content(), groundhogDay);
        embeddingStore.add(embeddingModel.embed(dieHard).content(), dieHard);

        final ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .dynamicFilter(sqlFilterBuilder::build) // LLM will generate the filter dynamically
                .build();

        final Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(chatModel)               // It should use OpenAI LLM
                .contentRetriever(contentRetriever)
                .build();

        final String question = "Recommend a good action movie from the 90s.";
        final String answer = assistant.chat(question);

        this.logger.info("Question: {}", question);
        this.logger.info("Answer  : {}", answer);

        if (this.logger.isTraceEnabled()) {
            this.logger.trace(exit());
        }
    }

    /// The user question record.
    ///
    /// @param userId   java.lang.String
    /// @param question java.lang.String
    record UserQuestion(
            String userId,
            String question
    ) {}
}
