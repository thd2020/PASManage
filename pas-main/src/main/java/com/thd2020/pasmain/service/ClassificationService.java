package com.thd2020.pasmain.service;

import com.thd2020.pasmain.dto.ClassificationResult;
import org.springframework.stereotype.Service;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

@Service
public class ClassificationService {
    
    private final Path pythonScriptPath = Paths.get("/home/lmj/xyx/PASManage/pas-main/src/main/resources/classify.py");
    private final String PYTHON_BINARY_PATH = "/home/lmj/anaconda3/envs/med_sam/bin/python";

    public ClassificationResult classifyImage(String imagePath) throws IOException, InterruptedException {
        ProcessBuilder processBuilder = new ProcessBuilder(
            PYTHON_BINARY_PATH,
            pythonScriptPath.toString(),
            "-img_path", imagePath
        );
        
        Process process = processBuilder.start();
        BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
        
        
        ClassificationResult result = new ClassificationResult();
        Map<String, Double> probabilities = new HashMap<>();
        
        String line;
        while ((line = reader.readLine()) != null) {
            if (line.startsWith("probabilities:")) {
                String[] parts = line.substring("probabilities:".length()).trim().split(",");
                for (String part : parts) {
                    String[] keyValue = part.trim().split("\\s+");
                    probabilities.put(keyValue[0].replace(":", ""), Double.parseDouble(keyValue[1]));
                }
                result.setProbabilities(probabilities);
            } else if (line.startsWith("predict type:")) {
                result.setPredictedType(line.substring("predict type:".length()).trim());
            }
        }

        int exitCode = process.waitFor();
        
        // Capture error message from stderr
        BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
        StringBuilder errorMessage = new StringBuilder();
        while ((line = errorReader.readLine()) != null) {
            errorMessage.append(line).append(System.lineSeparator());
        }

        // If the exit code is non-zero, throw a RuntimeException with the concrete error message
        if (exitCode != 0) {
            String errorOutput = errorMessage.toString().trim();
            if (errorOutput.isEmpty()) {
                errorOutput = "No additional error message provided by the process.";
            }
            throw new RuntimeException("Classification failed with exit code: " + exitCode + ". Error: " + errorOutput);
        }
        return result;
    }
}