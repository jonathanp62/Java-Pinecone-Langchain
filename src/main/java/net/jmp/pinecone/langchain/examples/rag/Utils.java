package net.jmp.pinecone.langchain.examples.rag;

/*
 * (#)Utils.java    0.1.0   05/30/2025
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

import java.io.IOException;

import java.net.URISyntaxException;
import java.net.URL;

import java.nio.file.*;

/// The utilities class.
///
/// @version    0.1.0
/// @since      0.1.0
public class Utils {
    /// The default constructor.
    private Utils() {
        super();
    }

    /// Glob a path.
    ///
    /// @param  glob    java.lang.String
    /// @return         java.nio.file.PathMatcher
    public static PathMatcher glob(final String glob) {
        return FileSystems.getDefault().getPathMatcher("glob:" + glob);
    }

    /// Convert a relative path to an absolute path.
    ///
    /// @param  relativePath  java.lang.String
    /// @return               java.nio.file.Path
    public static Path toPath(final String relativePath) {
        Path result = null;

        try {
            final URL fileUrl = Utils.class.getClassLoader().getResource(relativePath);

            if (fileUrl != null) {
                result = Paths.get(fileUrl.toURI());
            }
        } catch (final URISyntaxException e) {
            throw new RuntimeException(e);
        }

        return result;
    }

    /// Get the API key.
    ///
    /// @param  fileName    java.lang.String
    /// @return             java.util.Optional<java.lang.String>
    public static String getApiKey(final String fileName) {
        String apiKey;

        try {
            apiKey = Files.readString(Paths.get(fileName)).trim();
        } catch (final IOException ioe) {
            throw new RuntimeException(ioe);
        }

        return apiKey;
    }
}
