package com.thd2020.pasmain.service;

import ai.onnxruntime.*;
import org.opencv.core.Core;
import org.opencv.imgcodecs.Imgcodecs;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.thd2020.pasmain.repository.ImageRepository;
import com.thd2020.pasmain.repository.ImagingRecordRepository;
import com.thd2020.pasmain.util.ResourceUtils;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.List;
import org.apache.commons.io.FilenameUtils;

import javax.imageio.ImageIO;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import static org.bouncycastle.asn1.x500.style.RFC4519Style.c;

@Service
public class SegmentService {

    @Autowired
    private ImagingRecordRepository imagingRecordRepository;

    @Autowired
    private ImageRepository imageRepository;

    @Value("${resources.root}")
    private String LocalResRoot = "/home/lmj/xyx/pas";
    @Value("${python.path}")
    private String pythonPath = "/home/lmj/anaconda3/envs/med_sam/bin/python";

    Path tempDir, pythonScriptPath, encoderModelPath, decoderModelPath, multiSegmentScriptPath, multiSegmentModelPath;
    
    private final Path rootLocation = Paths.get(LocalResRoot);

    public SegmentService() throws OrtException, IOException {
        this.tempDir = Paths.get("/home/lmj/xyx/sda2/pasres");
        this.pythonScriptPath = ResourceUtils.extractResource("/segment.py", this.tempDir);
        this.encoderModelPath = ResourceUtils.extractResource("/sam-placenta.encoder.onnx", this.tempDir);
        this.decoderModelPath = ResourceUtils.extractResource("/sam-placenta.decoder.onnx", this.tempDir);
        this.multiSegmentScriptPath = ResourceUtils.extractResource("/multisegment.py", this.tempDir);
        this.multiSegmentModelPath = ResourceUtils.extractResource("/smsam.pth", this.tempDir);
    }


    public String segmentImagePy(String patientId, String recordId, String imagePath, String segmentationType, Map<String, Object> coordinates) throws IOException, InterruptedException {
        File savedImage = new File(imagePath);
        String imageName = FilenameUtils.removeExtension(savedImage.getName());
        String imagingRecordPath = imagingRecordRepository.findById(recordId).get().getPath();
        String outputDirPath = String.format("%s/masks/", imagingRecordPath);
        File outputDir = new File(outputDirPath);
        Files.createDirectories(outputDir.toPath());
        Path outputPath = Paths.get(outputDirPath, String.format("%s_mask.jpg", imageName));
        // 构建命令行参数
        ProcessBuilder pb = new ProcessBuilder(pythonPath, pythonScriptPath.toString(),
                "--encoder_model", encoderModelPath.toString(),
                "--decoder_model", decoderModelPath.toString(),
                "--img_path", imagePath,
                "--work_dir", outputDirPath);
        if (segmentationType.equalsIgnoreCase("POINT")) {
            String pointCoords = (String) coordinates.get("point_coords");
            String pointLabels = (String) coordinates.get("point_labels");
            pb.command().add("--point_coords");
            pb.command().add(pointCoords);
            pb.command().add("--point_labels");
            pb.command().add(pointLabels);
        } else if (segmentationType.equalsIgnoreCase("BOX")) {
            String boxes = (String) coordinates.get("boxes");
            pb.command().add("--boxes");
            pb.command().add(boxes);
        }

        // 启动Python进程
        Process process = pb.start();
        int exitCode = process.waitFor();

        // 返回分割后的图像路径
        return outputPath.toString();
    }

    public Map<String, String> multiSegmentImagePy(String patientId, String recordId, String imagePath, String promptType, List<String> targets, Map<String, Object> prompts) throws IOException, InterruptedException {
        File savedImage = new File(imagePath);
        String imageName = FilenameUtils.removeExtension(savedImage.getName());
        String outputDirPath = String.format("%s/%s/%s/masks/", rootLocation, patientId, recordId);
        File outputDir = new File(outputDirPath);
        Files.createDirectories(outputDir.toPath());
        String promptsJson;
        
        // Convert prompts to JSON 
        ObjectMapper mapper = new ObjectMapper();
        if (prompts == null) {
            promptsJson = mapper.writeValueAsString(prompts);
        }
        else {
            promptsJson = null;
        }

        // Build command
        ProcessBuilder pb = new ProcessBuilder(
            pythonPath,
            multiSegmentScriptPath.toString(),
            "--model_path", multiSegmentModelPath.toString(),
            "--img_path", imagePath,
            "--work_dir", outputDirPath
        );
        
        pb.command().add("--targets");

        // Add each target as a separate argument
        for (String target : targets) {
            pb.command().add(target);
        }

        if (promptType != null) {
            pb.command().add("--prompt_type");
            pb.command().add(promptType);
        }

        if (promptsJson != null) {
            pb.command().add("--prompts");
            pb.command().add(promptsJson);
        }

        // Run process
        Process process = pb.start();
        int exitCode = process.waitFor();
        String line;


        // If the exit code is non-zero, throw a RuntimeException with the concrete error message
        if (exitCode != 0) {
            // Capture error message from stderr
            BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
            StringBuilder errorMessage = new StringBuilder();
            while ((line = errorReader.readLine()) != null) {
                errorMessage.append(line).append(System.lineSeparator());
            }
            String errorOutput = errorMessage.toString().trim();
            if (errorOutput.isEmpty()) {
                errorOutput = "No additional error message provided by the process.";
            }
            throw new RuntimeException("Classification failed with exit code: " + exitCode + ". Error: " + errorOutput);
        }

        // Return paths to all generated masks
        Map<String, String> maskPaths = new HashMap<>();
        for (String target : targets) {
            String maskPath = Paths.get(outputDirPath, String.format("%s_%s_mask.jpg", imageName, target)).toString();
            maskPaths.put(target, maskPath);
        }

        return maskPaths;
    }
}
