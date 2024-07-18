package com.thd2020.pasmain.service;

import ai.onnxruntime.*;
import org.springframework.stereotype.Service;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import org.apache.commons.io.FilenameUtils;

import javax.imageio.ImageIO;

@Service
public class SegmentService {

    private static final Path rootLocation = Paths.get("/", "home",  "thd2020", "pas");

    private static final String ENCODER_MODEL_PATH = Paths.get(rootLocation.toString(), "models", "sam-placenta.encoder.onnx").toString();
    private static final String DECODER_MODEL_PATH = Paths.get(rootLocation.toString(), "models", "sam-placenta.decoder.onnx").toString();
    private static final String OUTPUT_DIR = "output";
    private static final String PYTHON_SCRIPT_PATH = "path/to/main.py";
    private static final String WORKDIR = "workdir";

    private final OrtEnvironment env;
    private final OrtSession encoderSession;
    private final OrtSession decoderSession;

    public SegmentService() throws OrtException, IOException {
        env = OrtEnvironment.getEnvironment();
        encoderSession = env.createSession(ENCODER_MODEL_PATH, new OrtSession.SessionOptions());
        decoderSession = env.createSession(DECODER_MODEL_PATH, new OrtSession.SessionOptions());
    }

    private float[][][][] convertImageToFloatTensor(BufferedImage image) {
        int targetHeight = 256;
        int targetWidth = 256;

        // Resize the image to the target dimensions
        Image resizedImage = image.getScaledInstance(targetWidth, targetHeight, Image.SCALE_SMOOTH);
        BufferedImage bufferedResizedImage = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = bufferedResizedImage.createGraphics();
        g2d.drawImage(resizedImage, 0, 0, null);
        g2d.dispose();

        float[][][][] result = new float[1][3][targetHeight][targetWidth];

        for (int y = 0; y < targetHeight; y++) {
            for (int x = 0; x < targetWidth; x++) {
                int pixel = bufferedResizedImage.getRGB(x, y);
                result[0][0][y][x] = ((pixel >> 16) & 0xFF); // Red channel
                result[0][1][y][x] = ((pixel >> 8) & 0xFF);  // Green channel
                result[0][2][y][x] = (pixel & 0xFF);         // Blue channel
            }
        }
        return result;
    }

    public String segmentImage(String patientId, String recordId, String imagePath, String segmentationType, Map<String, Object> coordinates) throws OrtException, IOException {
        // Load image
        BufferedImage image = ImageIO.read(new File(imagePath));
        float[][][][] imageData = convertImageToFloatTensor(image);
        float[][][] test = imageData[0];
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, imageData);

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
            float[][] defaultPointCoords = { { 256 / 2f, 256 / 2f } };
            float[] defaultPointLabels = { 1 };
            OnnxTensor pointCoordsTensor = OnnxTensor.createTensor(env, new float[][][] { defaultPointCoords });
            OnnxTensor pointLabelsTensor = OnnxTensor.createTensor(env, new float[][] { defaultPointLabels });
            decoderResult = decoderSession.run(Map.of(
                    "image_embeddings", imgEmbeddings,
                    "point_coords", pointCoordsTensor,
                    "point_labels", pointLabelsTensor,
                    "mask_input", OnnxTensor.createTensor(env, new float[1][1][64][64]),
                    "has_mask_input", OnnxTensor.createTensor(env, new float[] { 0 }),
                    "orig_im_size", OnnxTensor.createTensor(env, new float[] { image.getHeight(), image.getWidth() })
            ));
        }

        // Save output image
        float[][][][] masks4D = (float[][][][]) decoderResult.get(0).getValue();
        float[][] mask2D = masks4D[0][0]; // 取第一张2D mask


        // 应用sigmoid函数
        float[][] sigmoidMask = new float[mask2D.length][mask2D[0].length];
        for (int i = 0; i < mask2D.length; i++) {
            for (int j = 0; j < mask2D[i].length; j++) {
                sigmoidMask[i][j] = 0.5f * (float)Math.tanh(0.5f * mask2D[i][j]) + 1;
                sigmoidMask[i][j] = (sigmoidMask[i][j] > 0.5) ? 1.0f : 0.0f; // 应用mask threshold
            }
        }

        // 将2D mask转换为二值图像并保存
        BufferedImage maskImage = new BufferedImage(sigmoidMask[0].length, sigmoidMask.length, BufferedImage.TYPE_BYTE_BINARY);
        for (int i = 0; i < sigmoidMask.length; i++) {
            for (int j = 0; j < sigmoidMask[i].length; j++) {
                int pixelValue = (sigmoidMask[i][j] > 0.5) ? 255 : 0; // 将float转换为0或255
                maskImage.setRGB(j, i, pixelValue);
            }
        }


        // 保存图像
        File savedImage = new File(imagePath);
        String imageName = FilenameUtils.removeExtension(savedImage.getName());
        String outputDirPath = String.format("%s/%s/%s/masks/", rootLocation, patientId, recordId);
        File outputDir = new File(outputDirPath);
        Files.createDirectories(outputDir.toPath());
        Path outputPath = Paths.get(outputDirPath, String.format("%s_output.jpg", imageName));
        ImageIO.write(maskImage, "jpg", outputPath.toFile());

        return outputPath.toString();
    }

    public String segmentImagePy(String patientId, String recordId, String imagePath, String segmentationType, Map<String, Object> coordinates) throws IOException, InterruptedException {
        // 构建命令行参数
        ProcessBuilder pb = new ProcessBuilder("python3", PYTHON_SCRIPT_PATH,
                "--encoder_model", ENCODER_MODEL_PATH,
                "--decoder_model", DECODER_MODEL_PATH,
                "--img_path", imagePath,
                "--work_dir", WORKDIR);

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

        // 获取进程输出
        BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
        StringBuilder output = new StringBuilder();
        String line;
        while ((line = reader.readLine()) != null) {
            output.append(line).append("\n");
        }

        int exitCode = process.waitFor();
        if (exitCode != 0) {
            throw new IOException("Python script execution failed with exit code " + exitCode);
        }

        // 返回分割后的图像路径
        return output.toString().trim();
    }

    public void close() throws OrtException {
        encoderSession.close();
        decoderSession.close();
        env.close();
    }
}
