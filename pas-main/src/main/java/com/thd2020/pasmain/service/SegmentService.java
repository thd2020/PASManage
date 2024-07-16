package com.thd2020.pasmain.service;

import ai.onnxruntime.*;
import org.springframework.stereotype.Service;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import org.apache.commons.io.FilenameUtils;

@Service
public class SegmentService {

    private static final Path rootLocation = Paths.get("/", "home",  "thd2020", "pas");

    private static final String ENCODER_MODEL_PATH = Paths.get(rootLocation.toString(), "models", "sam-placenta.encoder.onnx").toString();
    private static final String DECODER_MODEL_PATH = Paths.get(rootLocation.toString(), "models", "sam-placenta.decoder.onnx").toString();
    private static final String OUTPUT_DIR = "output";

    private final OrtEnvironment env;
    private final OrtSession encoderSession;
    private final OrtSession decoderSession;

    public SegmentService() throws OrtException, IOException {
        env = OrtEnvironment.getEnvironment();
        encoderSession = env.createSession(ENCODER_MODEL_PATH, new OrtSession.SessionOptions());
        decoderSession = env.createSession(DECODER_MODEL_PATH, new OrtSession.SessionOptions());
    }

    public String segmentImage(String patientId, String recordId, String imagePath, String segmentationType, Map<String, Object> coordinates) throws OrtException, IOException {
        // Load image
        byte[] imageBytes = Files.readAllBytes(Paths.get(imagePath));
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, imageBytes);

        // Run encoder
        OrtSession.Result encoderResult = encoderSession.run(Map.of("input_image", inputTensor));
        OnnxTensor imgEmbeddings = (OnnxTensor) encoderResult.get(0);

        // Prepare decoder inputs based on segmentation type
        OrtSession.Result decoderResult;
        if ("POINT".equals(segmentationType)) {
            float[][] pointCoords = (float[][]) coordinates.get("pointCoords");
            float[] pointLabels = (float[]) coordinates.get("pointLabels");
            decoderResult = decoderSession.run(Map.of(
                    "img_embeddings", imgEmbeddings,
                    "point_coords", OnnxTensor.createTensor(env, pointCoords),
                    "point_labels", OnnxTensor.createTensor(env, pointLabels)
            ));
        } else if ("BOX".equals(segmentationType)) {
            float[][] boxes = (float[][]) coordinates.get("boxes");
            decoderResult = decoderSession.run(Map.of(
                    "img_embeddings", imgEmbeddings,
                    "boxes", OnnxTensor.createTensor(env, boxes)
            ));
        } else {
            decoderResult = decoderSession.run(Map.of(
                    "img_embeddings", imgEmbeddings
            ));
        }

        // Save output image
        float[][] masks = (float[][]) decoderResult.get(0).getValue();
        File image = new File(imagePath);
        String imageName = FilenameUtils.removeExtension(image.getName());
        Path outputPath = saveOutputImage(imageName, patientId, recordId, masks);

        return outputPath.toString();
    }

    private Path saveOutputImage(String imageMame, String patientId, String recordId, float[][] masks) throws IOException {
        String outputDirPath = String.format("%s/%s/%s/masks/", rootLocation, patientId, recordId);
        File outputDir = new File(outputDirPath);
        Files.createDirectories(outputDir.toPath());
        Path outputPath = Paths.get(outputDirPath ,String.format("%s_output.jpg", imageMame));
        // Implement your image saving logic here
        // Example: save the mask as an image file
        Files.write(outputPath, new byte[0]); // Replace with actual image saving logic
        return outputPath;
    }

    public void close() throws OrtException {
        encoderSession.close();
        decoderSession.close();
        env.close();
    }
}
