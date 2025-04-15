package com.thd2020.pasmain.util;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.*;

public class ResourceUtils {

    /**
     * Extracts a resource from the JAR into a real file in the provided output directory.
     * For example, to run Python files or load PyTorch checkpoints (.pth).
     *
     * @param resourcePath path inside the JAR (e.g., "/scripts/classify.py")
     * @param outputDir directory where the extracted file will be saved
     * @return the extracted file's absolute path on disk
     * @throws IOException if the resource is missing or fails to extract
     */
    public static Path extractResource(String resourcePath, Path outputDir) throws IOException {
        String fileName = Paths.get(resourcePath).getFileName().toString();
        Path outputPath = outputDir.resolve(fileName);
    
        // ✅ Avoid re-extracting if the file already exists
        if (Files.exists(outputPath)) {
            return outputPath;
        }
    
        try (InputStream is = ResourceUtils.class.getResourceAsStream(resourcePath)) {
            if (is == null) {
                throw new IOException("Resource not found: " + resourcePath);
            }
    
            Files.createDirectories(outputDir);
            Files.copy(is, outputPath, StandardCopyOption.REPLACE_EXISTING);
        }
    
        return outputPath;
    }    
}