package com.thd2020.pasmain.service;

import ai.onnxruntime.*;
import org.opencv.core.Core;
import org.opencv.imgcodecs.Imgcodecs;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import com.fasterxml.jackson.databind.ObjectMapper;

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

//import org.opencv.core.CvType;
//import org.opencv.core.Mat;
//import org.opencv.core.Size;
//import org.opencv.imgproc.Imgproc;

import javax.imageio.ImageIO;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import static org.bouncycastle.asn1.x500.style.RFC4519Style.c;

@Service
public class SegmentService {

    @Value("${resources.root}")
    private String LocalResRoot = "/home/lmj/xyx/pas";
    private String ResourcesRoot = "/home/lmj/xyx/PASManage/pas-main/src/main/resources";

    private final Path rootLocation = Paths.get(LocalResRoot);

    private final String ENCODER_MODEL_PATH = Paths.get(rootLocation.toString(), "models", "sam-placenta.encoder.onnx").toString();
    private final String DECODER_MODEL_PATH = Paths.get(rootLocation.toString(), "models", "sam-placenta.decoder.onnx").toString();
    private final String OUTPUT_DIR = "output";
    private final String PYTHON_SCRIPT_PATH = LocalResRoot+"/segment.py";
    private final String PYTHON_BINARY_PATH = "/home/lmj/anaconda3/envs/med_sam/bin/python";
    private final String WORKDIR = "workdir";
    private final String MULTI_SEGMENT_SCRIPT_PATH = ResourcesRoot+"/multisegment.py";
    private final String MULTI_SEGMENT_MODEL_PATH = "/home/lmj/xyx/ssam/logs/all_ep_60/Model/checkpoint_best.pth";

    private final OrtEnvironment env;
    private final OrtSession encoderSession;
    private final OrtSession decoderSession;


//    private final float[] pixelMean = {123.675f, 116.28f, 103.53f};
//    private final float[] pixelStd = {58.395f, 57.12f, 57.375f};
//    private final Size inputSize = new Size(256, 256);

    public SegmentService() throws OrtException, IOException {
        env = OrtEnvironment.getEnvironment();
        encoderSession = env.createSession(ENCODER_MODEL_PATH, new OrtSession.SessionOptions());
        decoderSession = env.createSession(DECODER_MODEL_PATH, new OrtSession.SessionOptions());
    }

/**    private float[][][][] convertImageToFloatTensor(BufferedImage image) {
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

    public float[][][][] transform(Mat img) {
        // BGR -> RGB
        Imgproc.cvtColor(img, img, Imgproc.COLOR_BGR2RGB);

        // Normalization
        for (int i = 0; i < img.rows(); i++) {
            for (int j = 0; j < img.cols(); j++) {
                double[] pixel = img.get(i, j);
                pixel[0] = (pixel[0] - pixelMean[0]) / pixelStd[0];
                pixel[1] = (pixel[1] - pixelMean[1]) / pixelStd[1];
                pixel[2] = (pixel[2] - pixelMean[2]) / pixelStd[2];
                img.put(i, j, pixel);
            }
        }

        // Resize
        Imgproc.resize(img, img, inputSize, 0, 0, Imgproc.INTER_NEAREST);

        // HWC -> CHW
        float[][][][] nchwImage = new float[1][3][(int) inputSize.height][(int) inputSize.width];
        for (int i = 0; i < inputSize.height; i++) {
            for (int j = 0; j < inputSize.width; j++) {
                double[] pixel = img.get(i, j);
                nchwImage[0][0][i][j] = (float) pixel[0];
                nchwImage[0][1][i][j] = (float) pixel[1];
                nchwImage[0][2][i][j] = (float) pixel[2];
            }
        }

        return nchwImage;
    }

    public String segmentImage(String patientId, String recordId, String imagePath, String segmentationType, Map<String, Object> coordinates) throws OrtException, IOException {
        // Load image
        BufferedImage image = ImageIO.read(new File(imagePath));
        //float[][][][] imageData = convertImageToFloatTensor(image);
        Mat img = Imgcodecs.imread(imagePath);
        float[][][][] imageData = transform(img);
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
        double[] flatArray = flattenArray(mask2D);

        // Calculate statistics
        DescriptiveStatistics stats = new DescriptiveStatistics(flatArray);

        System.out.println("Maximum: " + stats.getMax());
        System.out.println("Minimum: " + stats.getMin());
        System.out.println("Mean: " + stats.getMean());
        System.out.println("Standard Deviation: " + stats.getStandardDeviation());

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
    } **/

    private static double[] flattenArray(float[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[] flatArray = new double[rows * cols];
        int index = 0;

        for (float[] row : matrix) {
            for (float value : row) {
                flatArray[index++] = value;
            }
        }

        return flatArray;
    }

    public String segmentImagePy(String patientId, String recordId, String imagePath, String segmentationType, Map<String, Object> coordinates) throws IOException, InterruptedException {
        File savedImage = new File(imagePath);
        String imageName = FilenameUtils.removeExtension(savedImage.getName());
        String outputDirPath = String.format("%s/%s/%s/masks/", rootLocation, patientId, recordId);
        File outputDir = new File(outputDirPath);
        Files.createDirectories(outputDir.toPath());
        Path outputPath = Paths.get(outputDirPath, String.format("%s_mask.jpg", imageName));
        // 构建命令行参数
        ProcessBuilder pb = new ProcessBuilder(PYTHON_BINARY_PATH, PYTHON_SCRIPT_PATH,
                "--encoder_model", ENCODER_MODEL_PATH,
                "--decoder_model", DECODER_MODEL_PATH,
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
            PYTHON_BINARY_PATH,
            MULTI_SEGMENT_SCRIPT_PATH,
            "--model_path", MULTI_SEGMENT_MODEL_PATH,
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

    public void close() throws OrtException {
        encoderSession.close();
        decoderSession.close();
        env.close();
    }
}
