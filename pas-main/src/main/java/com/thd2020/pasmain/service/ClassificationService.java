package com.thd2020.pasmain.service;

import com.thd2020.pasmain.dto.ClassificationResult;
import com.thd2020.pasmain.entity.MedicalRecord;
import com.thd2020.pasmain.util.ResourceUtils;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

@Service
public class ClassificationService {

    Path tempDir, pythonScriptPath, mlmpasPath, mtpasPath, vgg16Path;
    Path vlmPthPath, lvmmedPthPath, mtpasPthPath, vggPthPath;

    @Value("${python.path}")
    private final String PYTHON_BINARY_PATH = "/home/lmj/anaconda3/envs/med_sam/bin/python";

    public ClassificationService() throws IOException {
        this.tempDir = Paths.get("/home/lmj/xyx/sda2/pasres");
        this.pythonScriptPath = ResourceUtils.extractResource("/classify.py", this.tempDir);
        this.mlmpasPath = ResourceUtils.extractResource("/MLMPAS.py", this.tempDir);
        this.mtpasPath = ResourceUtils.extractResource("/MTPAS.py", this.tempDir);
        this.vgg16Path = ResourceUtils.extractResource("/vgg16.py", this.tempDir);
    
        this.vlmPthPath = ResourceUtils.extractResource("/vlm.pth", this.tempDir);
        this.lvmmedPthPath = ResourceUtils.extractResource("/lvm-med.pth", this.tempDir);
        this.mtpasPthPath = ResourceUtils.extractResource("/MTPAS.py", this.tempDir);
        this.vggPthPath = ResourceUtils.extractResource("/vgg.pth", this.tempDir);
    }



    public ClassificationResult classifyImage(String imagePath) throws IOException, InterruptedException {
        ProcessBuilder processBuilder = new ProcessBuilder(
            PYTHON_BINARY_PATH,
            pythonScriptPath.toString(),
            "-ckpt_path", vlmPthPath.toString(),
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
        Path pthPath;
        switch(model.toLowerCase()) {
            case "mlmpas": scriptPath = mlmpasPath; pthPath = lvmmedPthPath; break;
            case "mtpas": scriptPath = mtpasPath; pthPath = mtpasPthPath; break;
            case "vgg16": scriptPath = vgg16Path; pthPath = vggPthPath; break;
            default: throw new IllegalArgumentException("Unknown model: " + model);
        }
        ProcessBuilder processBuilder = new ProcessBuilder(
            PYTHON_BINARY_PATH,
            scriptPath.toString(),
            "-img_path", imagePath,
            "-age", String.valueOf(age),
            "-placenta_previa", String.valueOf(placentaPrevia),
            "-c_section_count", String.valueOf(cSectionCount),
            "-had_abortion", String.valueOf(hadAbortion),
            "-ckpt_path", pthPath.toString()
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