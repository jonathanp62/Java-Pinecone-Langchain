package net.jmp.pinecone.langchain;

/*
 * (#)StructuredOutput.java 0.1.0   06/03/2025
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

import com.fasterxml.jackson.databind.ObjectMapper;

import dev.langchain4j.data.message.UserMessage;

import dev.langchain4j.model.chat.ChatModel;

import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ResponseFormat;

import static dev.langchain4j.model.chat.request.ResponseFormatType.JSON;

import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import dev.langchain4j.model.chat.request.json.JsonSchema;

import dev.langchain4j.model.chat.response.ChatResponse;

import dev.langchain4j.model.openai.OpenAiChatModel;

import static dev.langchain4j.model.openai.OpenAiChatModelName.GPT_4_1;

import net.jmp.pinecone.langchain.examples.rag.Utils;

import static net.jmp.util.logging.LoggerUtils.*;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/// The structured output class.
///
/// @version    0.1.0
/// @since      0.1.0
final class StructuredOutput extends Operation {
    /// The logger.
    private final Logger logger = LoggerFactory.getLogger(this.getClass().getName());

    /// The default constructor.
    StructuredOutput() {
        super();
    }

    /// The operate method.
    @Override
    void operate() {
        if (this.logger.isTraceEnabled()) {
            this.logger.trace(entry());
        }

        final String openaiApiKey = System.getProperty("app.openaiApiKey");

        this.logger.info("Structured Output");

        if (this.logger.isDebugEnabled()) {
            this.logger.debug("OpenAI Api Key  : {}", openaiApiKey);
        }

        final ResponseFormat responseFormat = ResponseFormat.builder()
                .type(JSON) // Type can be either TEXT (default) or JSON
                .jsonSchema(JsonSchema.builder()
                        .name("Person") // OpenAI requires specifying the name for the schema
                        .rootElement(JsonObjectSchema.builder()
                                .addStringProperty("name")
                                .addIntegerProperty("age")
                                .addNumberProperty("height")
                                .addBooleanProperty("married")
                                .required("name", "age", "height", "married")
                                .build())
                        .build())
                .build();

        final UserMessage userMessage = UserMessage.from("""
                John is 42 years old and lives an independent life.
                He stands 1.75 meters tall and carries himself with confidence.
                Currently unmarried, he enjoys the freedom to focus on his personal goals and interests.
                """);

        final ChatRequest chatRequest = ChatRequest.builder()
                .responseFormat(responseFormat)
                .messages(userMessage)
                .build();

        final ChatModel chatModel = OpenAiChatModel.builder()
                .apiKey(Utils.getApiKey(openaiApiKey))
                .modelName(GPT_4_1)
                .logRequests(true)
                .logResponses(true)
                .build();

        final ChatResponse chatResponse = chatModel.chat(chatRequest);
        final String output = chatResponse.aiMessage().text();

        Person person = null;

        try {
            person = new ObjectMapper().readValue(output, Person.class);
        } catch (Exception e) {
            this.logger.error(catching(e));
        }

        this.logger.info("Output: {}", output);

        if (person != null) {
            this.logger.info("Person: {}", person);
        }

        if (this.logger.isTraceEnabled()) {
            this.logger.trace(exit());
        }
    }

    /// The person record.
    ///
    /// @param name     java.lang.String
    /// @param age      int
    /// @param height   double
    /// @param married  boolean
    record Person(String name, int age, double height, boolean married) {
    }
}
