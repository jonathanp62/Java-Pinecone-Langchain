package net.jmp.pinecone.langchain;

/*
 * (#)Main.java 0.1.0   05/28/2025
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

import net.jmp.pinecone.langchain.examples.rag.*;

import static net.jmp.util.logging.LoggerUtils.*;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/// The main application class.
///
/// @version    0.1.0
/// @since      0.1.0
public final class Main implements Runnable {
    /// The logger.
    private final Logger logger = LoggerFactory.getLogger(this.getClass().getName());

    /// The default constructor.
    private Main() {
        super();
    }

    /// The run method.
    @Override
    public void run() {
        if (this.logger.isTraceEnabled()) {
            this.logger.trace(entry());
        }

        final String operation = System.getProperty("app.operation");

        this.logger.info("Pinecone Langchain Operation: {}", operation);

        switch (operation) {
            case "delete":
                new Delete().operate();
                break;
            case "easyrag":
                new EasyRag().run();
                break;
            case "load":
                new Load().operate();
                break;
            case "metadatarag":
                new MetadataRag().run();
                break;
            case "naiverag":
                new NaiveRag().run();
                break;
            case "query":
                new Query().operate();
                break;
            case "querycompressionrag":
                new QueryCompressionRag().run();
                break;
            case "queryroutingrag":
                new QueryRoutingRag().run();
                break;
            case "rag":
                new Rag().operate();
                break;
            case "rerankingrag":
                new RerankingRag().run();
                break;
            default:
                this.logger.error("Unknown operation: {}", operation);
        }

        if (this.logger.isTraceEnabled()) {
            this.logger.trace(exit());
        }
    }

    /// The main application entry point.
    ///
    /// @param  args    java.lang.String[]
    public static void main(String[] args) {
        new Main().run();
    }
}