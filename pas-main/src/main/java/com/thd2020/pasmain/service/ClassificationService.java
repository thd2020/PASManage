package com.thd2020.pasmain.service;

import com.thd2020.pasmain.dto.ClassificationResult;
import com.thd2020.pasmain.entity.MedicalRecord;

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
    
    private final Path mlmpasPath = Paths.get("/home/lmj/xyx/PASManage/pas-main/src/main/resources/MLMPAS.py");
    private final Path mtpasPath = Paths.get("/home/lmj/xyx/PASManage/pas-main/src/main/resources/MTPAS.py");
    private final Path vgg16Path = Paths.get("/home/lmj/xyx/PASManage/pas-main/src/main/resources/vgg16.py");

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
    
    public ClassificationResult multiModalClassify(
        String imagePath, 
        int age,
        int placentaPrevia,
        int cSectionCount,
        int hadAbortion,
        String model) throws IOException, InterruptedException {
        Path scriptPath;
        switch(model.toLowerCase()) {
            case "mlmpas": scriptPath = mlmpasPath; break;
            case "mtpas": scriptPath = mtpasPath; break;
            case "vgg16": scriptPath = vgg16Path; break;
            default: throw new IllegalArgumentException("Unknown model: " + model);
        }
        ProcessBuilder processBuilder = new ProcessBuilder(
            PYTHON_BINARY_PATH,
            scriptPath.toString(),
            "-img_path", imagePath,
            "-age", String.valueOf(age),
            "-placenta_previa", String.valueOf(placentaPrevia),
            "-c_section_count", String.valueOf(cSectionCount),
            "-had_abortion", String.valueOf(hadAbortion)
        );
        return executeClassification(processBuilder);
    }
    private ClassificationResult executeClassification(ProcessBuilder processBuilder) 
            throws IOException, InterruptedException {
        Process process = processBuilder.start();
        BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
        
        ClassificationResult result = new ClassificationResult();
        Map<String, Double> probabilities = new HashMap<>();
        
        String line;
        while ((line = reader.readLine()) != null) {
            if (line.startsWith("probabilities:")) {
                String[] parts = line.substring("probabilities:".length()).trim().split(",");
                for (String part : parts) {
                    String[] keyValue = part.trim().split(":\\s+");
                    probabilities.put(keyValue[0], Double.parseDouble(keyValue[1]));
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