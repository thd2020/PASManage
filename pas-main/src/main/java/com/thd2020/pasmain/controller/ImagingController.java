package com.thd2020.pasmain.controller;

import ai.onnxruntime.OrtException;
import com.thd2020.pasmain.dto.ApiResponse;
import com.thd2020.pasmain.dto.ClassificationResult;
import com.thd2020.pasmain.entity.*;
import com.thd2020.pasmain.repository.ImageRepository;
import com.thd2020.pasmain.repository.ImagingRecordRepository;
import com.thd2020.pasmain.repository.MaskRepository;
import com.thd2020.pasmain.repository.PlacentaSegmentationGradingRepository;
import com.thd2020.pasmain.service.ClassificationService;
import com.thd2020.pasmain.service.ImagingService;
import com.thd2020.pasmain.service.PRInfoService;
import com.thd2020.pasmain.service.PatientService;
import com.thd2020.pasmain.service.SegmentService;
import com.thd2020.pasmain.service.UserService;
import com.thd2020.pasmain.util.JwtUtil;
import com.thd2020.pasmain.util.UtilFunctions;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;
import org.springframework.core.io.UrlResource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.net.MalformedURLException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

@RestController
@RequestMapping("/api/v1/imaging")
@Tag(name = "Imaging Controller", description = "影像模块接口")
public class ImagingController {

    @Autowired
    private ImagingService imagingService;

    @Autowired
    private PatientService patientService;

    @Autowired
    private SegmentService segmentService;

    @Autowired
    private JwtUtil jwtUtil;

    @Autowired
    private UtilFunctions utilFunctions;

    @Autowired
    private ImagingRecordRepository imagingRecordRepository;

    @Autowired
    private ImageRepository imageRepository;

    @Autowired
    private MaskRepository maskRepository;

    @Autowired
    private PlacentaSegmentationGradingRepository placentaSegmentationGradingRepository;

    @Autowired
    private UserService userService;

    @Autowired
    private ClassificationService classificationService;

    @Autowired
    private PRInfoService prInfoService;

    @PostMapping(value = "/segment-image", consumes= MediaType.MULTIPART_FORM_DATA_VALUE)
    @Operation(summary = "图像分割", description = "基于点或框的提示进行图像分割")
    public ResponseEntity<?> segmentImage(
            @Parameter(description = "影像记录ID", required = true) @RequestParam String recordId,
            @Parameter(description = "图像文件", required = true) @RequestPart("file") MultipartFile image,
            @Parameter(description = "分割提示类型", required = true) @RequestParam String hintType,
            @Parameter(description = "提示坐标", required = false) @RequestParam Map<String, Object> hintCoordinates,
            @RequestHeader("Authorization") String token) throws IOException, OrtException, InterruptedException {
        String patientId = imagingRecordRepository.findById(recordId).get().getPatient().getPatientId();
        if (!utilFunctions.isAdmin(token) && !utilFunctions.isDoctor(token) && !utilFunctions.isMatch(token, patientService.getPatient(patientId).getUser().getUserId())) {
            return ResponseEntity
                    .status(401)
                    .build();
        }
        // Step 1: 添加图像记录
        Image savedImage = imagingService.addImage(recordId, image);
        if (savedImage == null) {
            return ResponseEntity
                    .status(500)
                    .body("Failed to add image");
        }
        Long imageId = savedImage.getImageId();
        // Step 2: 图像分割
        String segmentedImagePath = segmentService.segmentImagePy(
                patientId.toString(),
                recordId,
                savedImage.getImagePath(),
                hintType,
                hintCoordinates
        );
        if (segmentedImagePath == null) {
            return ResponseEntity
                    .status(404)
                    .body("Failed to segment image, result is null");
        }
        Mask addedMask = imagingService.addMask(imageId, Paths.get(segmentedImagePath), "MODEL");
        if (addedMask == null) {
            return ResponseEntity
                    .status(404)
                    .body("Failed to add mask");
        }
        FileSystemResource resultMask = new FileSystemResource(segmentedImagePath);
        return ResponseEntity.ok()
                .contentType(MediaType.IMAGE_JPEG)
                .header(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=\"" + addedMask.getSegmentationMaskPath() + "\"")
                .body(resultMask);
    }

    @PostMapping(value = "/segment-exist-image")
        @Operation(summary = "已有图像分割", description = "基于点或框的提示进行已有图像分割")
        public ResponseEntity<?> segmentExistingImage(
                @Parameter(description = "影像记录ID", required = true) @RequestParam String recordId,
                @Parameter(description = "图像ID", required = true) @RequestParam Long imageId,
                @Parameter(description = "分割提示类型", required = true) @RequestParam String hintType,
                @Parameter(description = "提示坐标", required = false) @RequestParam Map<String, Object> hintCoordinates,
                @RequestHeader("Authorization") String token) throws IOException, OrtException, InterruptedException {
            String patientId = imagingRecordRepository.findById(recordId).get().getPatient().getPatientId();
            if (!utilFunctions.isAdmin(token) && !utilFunctions.isDoctor(token) && !utilFunctions.isMatch(token, patientService.getPatient(patientId).getUser().getUserId())) {
                return ResponseEntity
                        .status(401)
                        .build();
            }
            // Step 1: 添加图像记录
            Image detectedImage = imagingService.getImage(imageId);
            if (detectedImage == null) {
                return ResponseEntity
                        .status(500)
                        .body("Failed to add image");
            }
            // Step 2: 图像分割
            String segmentedImagePath = segmentService.segmentImagePy(
                    patientId.toString(),
                    recordId,
                    detectedImage.getImagePath(),
                    hintType,
                    hintCoordinates
            );
            if (segmentedImagePath == null) {
                return ResponseEntity
                        .status(404)
                        .body("Failed to segment image, result is null");
            }
            Mask addedMask = imagingService.addMask(imageId, Paths.get(segmentedImagePath), "MODEL");
            if (addedMask == null) {
                return ResponseEntity
                        .status(404)
                        .body("Failed to add mask");
            }
            FileSystemResource resultMask = new FileSystemResource(segmentedImagePath);
            return ResponseEntity.ok()
                    .contentType(MediaType.IMAGE_JPEG)
                    .header(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=\"" + addedMask.getSegmentationMaskPath() + "\"")
                    .body(resultMask);
        }

    @PostMapping(value = "/classify-image", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    @Operation(summary = "图像分类", description = "对上传的图像进行分类")
    public ResponseEntity<?> classifyImage(
            @Parameter(description = "影像记录ID", required = true) @RequestParam String recordId,
            @Parameter(description = "图像文件", required = true) @RequestPart("file") MultipartFile image,
            @RequestHeader("Authorization") String token) throws IOException, InterruptedException {
        String patientId = imagingRecordRepository.findById(recordId).get().getPatient().getPatientId();
        if (!utilFunctions.isAdmin(token) && !utilFunctions.isDoctor(token)) {
            return ResponseEntity.status(401).build();
        }

        Image savedImage = imagingService.addImage(recordId, image);
        if (savedImage == null) {
            return ResponseEntity.status(500).body("Failed to add image");
        }

        ClassificationResult classificationResult = classificationService.classifyImage(savedImage.getImagePath());

        PlacentaClassificationResult result = imagingService.addClassification(
            savedImage.getImageId(),
            classificationResult,
            "RESNET"
        );

        if (result == null) {
            return ResponseEntity.status(500).body("Failed to save classification result");
        }

        return ResponseEntity.ok(result);
    }

    @PostMapping("/classify-exist-image")
    @Operation(summary = "图像分类", description = "对已存在的图像进行分类")
    public ResponseEntity<?> classifyExistingImage(
        @Parameter(description = "影像记录ID", required = true) @RequestParam String recordId,
        @Parameter(description = "图像ID", required = true) @RequestParam Long imageId,
        @RequestHeader("Authorization") String token) throws IOException, InterruptedException {
        String patientId = imagingRecordRepository.findById(recordId).get().getPatient().getPatientId();
        if (!utilFunctions.isAdmin(token) && !utilFunctions.isDoctor(token)) {
        return ResponseEntity.status(401).build();
        }

        Image existingImage = imagingService.getImage(imageId);
        if (existingImage == null) {
        return ResponseEntity.status(404).body("Image not found");
        }

        ClassificationResult classificationResult = classificationService.classifyImage(existingImage.getImagePath());

        PlacentaClassificationResult result = imagingService.addClassification(
        existingImage.getImageId(),
        classificationResult,
        "RESNET"
        );

        if (result == null) {
        return ResponseEntity.status(500).body("Failed to save classification result");
        }

        return ResponseEntity.ok(result);
    }
    @Operation(summary = "添加影像记录", description = "管理员和医生可以添加影像记录")
    @PostMapping("/records")
    public ApiResponse<?> addImagingRecord(
            @Parameter(description = "影像记录实体", required = true) @RequestBody ImagingRecord imagingRecord,
            @RequestHeader("Authorization") String token) {
        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token)) {
            ImagingRecord addedImagingRecord = imagingService.addImagingRecord(imagingRecord);
            return new ApiResponse<>(addedImagingRecord!=null?"success":"failure", addedImagingRecord!=null?"Successfully adding imaging record":"failed to add imaging record", addedImagingRecord);
        } else {
            return new ApiResponse<>("error", "Unauthorized", null);
        }
    }

    @Operation(summary = "获取影像记录", description = "获取影像记录及其相关图像")
    @GetMapping("/records/{recordId}")
    public ApiResponse<ImagingRecord> getImagingRecord(
            @Parameter(description = "影像记录ID", required = true) @PathVariable String recordId,
            @RequestHeader("Authorization") String token) {
        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token) || utilFunctions.isMatch(token, imagingRecordRepository.getReferenceById(recordId).getPatient().getUser().getUserId())) {
            ImagingRecord gottenImagingRecord = imagingService.getImagingRecord(recordId);
            return new ApiResponse<>(gottenImagingRecord!=null?"success":"failure", gottenImagingRecord!=null?"Successfully got imaging record":"failed to get imaging record", gottenImagingRecord);
        } else {
            return new ApiResponse<>("error", "Unauthorized", null);
        }
    }

    @Operation(summary = "更新影像记录", description = "管理员和医生可以更新影像记录")
    @PutMapping("/records/{recordId}")
    public ApiResponse<ImagingRecord> updateImagingRecord(
            @Parameter(description = "影像记录ID", required = true) @PathVariable String recordId,
            @Parameter(description = "更新后的影像记录实体", required = true) @RequestBody ImagingRecord imagingRecord,
            @RequestHeader("Authorization") String token) {
        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token) || utilFunctions.isMatch(token, imagingRecordRepository.getReferenceById(recordId).getPatient().getUser().getUserId())) {
            ImagingRecord updatedImagingRecord = imagingService.updateImagingRecord(recordId, imagingRecord);
            return new ApiResponse<>(updatedImagingRecord!=null?"success":"failure", updatedImagingRecord!=null?"Successfully updated imaging record":"failed to update imaging record", updatedImagingRecord);
        } else {
            return new ApiResponse<>("error", "Unauthorized", null);
        }
    }

    @Operation(summary = "删除影像记录", description = "管理员和医生可以删除影像记录")
    @DeleteMapping("/records/{recordId}")
    public ApiResponse<?> deleteImagingRecord(
            @Parameter(description = "影像记录ID", required = true) @PathVariable String recordId,
            @RequestHeader("Authorization") String token) {
        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token)) {
            int status = imagingService.deleteImagingRecord(recordId);
            return new ApiResponse<>(status==0?"success":"failure", status==0?"Successfully deleted imaging record":"failed to delete imaging record", status);
        } else {
            return new ApiResponse<>("error", "Unauthorized", null);
        }
    }

    @Operation(summary = "添加图像", description = "管理员和医生可以添加图像")
    @PostMapping(value = "/images", consumes= MediaType.MULTIPART_FORM_DATA_VALUE)
    public ApiResponse<?> addImage(
            @Parameter(description = "要添加进的检测记录id", required = true) @RequestParam String recordId,
            @Parameter(description = "图像实体", required = true) @RequestPart("file") MultipartFile image,
            @RequestHeader("Authorization") String token) throws IOException {
        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token)) {
            Image addedImage = imagingService.addImage(recordId, image);
            return new ApiResponse<>("success", "Image added successfully", addedImage);
        } else {
            return new ApiResponse<>("error", "Unauthorized", null);
        }
    }

    @Operation(summary = "获取图像", description = "获取图像文件")
    @GetMapping("/images/{imageId}")
    public ResponseEntity<?> getImage(
            @Parameter(description = "图像ID", required = true) @PathVariable Long imageId,
            @RequestHeader("Authorization") String token) throws IOException {
        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token) || utilFunctions.isMatch(token, imageRepository.getReferenceById(imageId).getPatient().getUser().getUserId())) {
            Image image = imagingService.getImage(imageId);
            if (image != null && image.getImageAvail() != Image.Availability.NONEXIST) {
                return ResponseEntity.ok()
                        .contentType(MediaType.IMAGE_JPEG)
                        .header(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=\"" + image.getImagePath() + "\"")
                        .body(new FileSystemResource(image.getImagePath()));
            }
            else {
                return ResponseEntity.status(404).body("Cannot find image locally");
            }
        }
        return ResponseEntity.status(503).body("Unauthorised - Not Enough Privileges");
    }

    @Operation(summary = "获取图像信息", description = "获取图像相关信息")
    @GetMapping("/images/info/{imageId}")
    public ApiResponse<Image> updateImage(
            @Parameter(description = "图像ID", required = true) @PathVariable Long imageId,
            @RequestHeader("Authorization") String token) throws MalformedURLException {
        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token) || utilFunctions.isMatch(token, imageId)) {
            Image image = imagingService.getImage(imageId);
            return new ApiResponse<>(image!=null?"success":"failure", image!=null?"Successfully got image":"failed to get image", image);
        } else {
            return new ApiResponse<>("error", "Unauthorized", null);
        }
    }

    @Operation(summary = "更新图像", description = "管理员和医生可以更新图像")
    @PutMapping("/images/{imageId}")
    public ApiResponse<Image> updateImage(
            @Parameter(description = "图像ID", required = true) @PathVariable Long imageId,
            @Parameter(description = "更新后的图像实体", required = true) @RequestBody Image image,
            @RequestHeader("Authorization") String token) {
        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token) || utilFunctions.isMatch(token, imageId)) {
            Image updatedImage = imagingService.updateImage(imageId, image);
            return new ApiResponse<>(updatedImage!=null?"success":"failure", updatedImage!=null?"Successfully updated image":"failed to update image", updatedImage);
        } else {
            return new ApiResponse<>("error", "Unauthorized", null);
        }
    }

    @Operation(summary = "删除图像", description = "管理员和医生可以删除图像")
    @DeleteMapping("/images/{imageId}")
    public ApiResponse<?> deleteImage(
            @Parameter(description = "图像ID", required = true) @PathVariable Long imageId,
            @RequestHeader("Authorization") String token) {
        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token)) {
            int status = imagingService.deleteImage(imageId);
            return new ApiResponse<>(status==0?"success":"failure", status==0?"Successfully deleted image":"failed to delete image", status);
        } else {
            return new ApiResponse<>("error", "Unauthorized", null);
        }
    }

    @Operation(summary = "添加掩膜", description = "管理员和医生可以添加掩膜")
    @PostMapping(value = "/masks", consumes= MediaType.MULTIPART_FORM_DATA_VALUE)
    public ApiResponse<?> addMask(
            @Parameter(description = "对应的图像id", required = true) @RequestParam Long imageId,
            @Parameter(description = "图像来源:DOCTOR or MODEL", required = true) @RequestParam String source,
            @Parameter(description = "掩膜实体", required = true)  @RequestPart("file") MultipartFile mask,
            @Parameter(description = "掩膜json实体", required = false)  @RequestPart("json_file") MultipartFile segmentationJson,
            @RequestHeader("Authorization") String token) throws IOException {
        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token)) {
            Mask addedMask = null;
            if (segmentationJson != null) {
                addedMask = imagingService.addMask(imageId, mask, segmentationJson, source);
            } else {
                addedMask = imagingService.addMask(imageId, mask, source);
            }
            return new ApiResponse<>("success", "Successfully added mask", addedMask);
        } else {
            return new ApiResponse<>("error", "Unauthorized", null);
        }
    }

    @Operation(summary = "获取掩膜", description = "获取掩膜文件")
    @GetMapping("/masks/{maskId}")
    public ResponseEntity<?> getMask(
            @Parameter(description = "掩膜ID", required = true) @PathVariable Long maskId,
            @RequestHeader("Authorization") String token) throws IOException {
        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token) || utilFunctions.isMatch(token, maskRepository.getReferenceById(maskId).getImage().getPatient().getUser().getUserId())) {
            Mask mask = imagingService.getMask(maskId);
            if (mask != null) {
                return ResponseEntity.ok()
                        .contentType(MediaType.IMAGE_JPEG)
                        .header(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=\"" + mask.getSegmentationMaskPath() + "\"")
                        .body(new FileSystemResource(mask.getSegmentationMaskPath()));
            }
        }
        return ResponseEntity.status(404)
            .body("No such mask file");
    }

    @Operation(summary = "获取掩膜", description = "获取掩膜详细信息")
    @GetMapping("/masks/info/{maskId}")
    public ApiResponse<?> getMaskInfo(
            @Parameter(description = "掩膜ID", required = true) @PathVariable Long maskId,
            @RequestHeader("Authorization") String token) throws MalformedURLException {
        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token) || utilFunctions.isMatch(token, maskId)) {
            return new ApiResponse<>("success", "Mask fetched successfully", imagingService.getMask(maskId));
        } else {
            return new ApiResponse<>("error", "Unauthorized", null);
        }
    }

    @Operation(summary = "更新掩膜", description = "管理员和医生可以更新掩膜")
    @PutMapping("/masks/{maskId}")
    public ApiResponse<Mask> updateMask(
            @Parameter(description = "掩膜ID", required = true) @PathVariable Long maskId,
            @Parameter(description = "更新后的掩膜实体", required = true) @RequestBody Mask mask,
            @RequestHeader("Authorization") String token) {
        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token) || utilFunctions.isMatch(token, maskId)) {
            return new ApiResponse<>("success", "Mask fetched successfully", imagingService.updateMask(maskId, mask));
        } else {
            return new ApiResponse<>("error", "Unauthorized", null);
        }
    }

    @Operation(summary = "删除掩膜", description = "管理员和医生可以删除掩膜")
    @DeleteMapping("/masks/{maskId}")
    public ApiResponse<?> deleteMask(
            @Parameter(description = "掩膜ID", required = true) @PathVariable Long maskId,
            @RequestHeader("Authorization") String token) {
        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token)) {
            int status = imagingService.deleteMask(maskId);
            return new ApiResponse<>(status==0?"success":"failure", status==0?"successfully deleted mask":"failed to delete mask", status);
        } else {
            return new ApiResponse<>("error", "Unauthorized", null);
        }
    }

    @Operation(summary = "添加分割/分级结果", description = "管理员和医生可以添加分割/分级结果")
    @PostMapping("/gradings")
    public ApiResponse<?> addGrading(
            @Parameter(description = "图像ID", required = true) @RequestParam Long imageId,
            @Parameter(description = "掩膜ID", required = true) @RequestParam Long maskId,
            @Parameter(description = "分割/分级结果实体", required = true) @RequestBody PlacentaSegmentationGrading grading,
            @RequestHeader("Authorization") String token) {
        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token)) {
            return new ApiResponse<>("success", "Successfully added sengmentation/grading result", imagingService.addGrading(imageId, maskId, grading));
        } else {
            return new ApiResponse<>("error", "Unauthorized", null);
        }
    }

    @Operation(summary = "获取分割/分级结果", description = "获取分割/分级结果及其相关图像和掩膜")
    @GetMapping("/gradings/{gradingId}")
    public ApiResponse<PlacentaSegmentationGrading> getGrading(
            @Parameter(description = "分割/分级结果ID", required = true) @PathVariable Long gradingId,
            @RequestHeader("Authorization") String token) throws IOException {
        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token) || utilFunctions.isMatch(token, placentaSegmentationGradingRepository.findById(gradingId).orElseThrow().getPatient().getUser().getUserId())) {
            return new ApiResponse<>("success", "Successfully got segmentation/grading result", imagingService.getGrading(gradingId));
        } else {
            return new ApiResponse<>("error", "Unauthorized", null);
        }
    }

    @Operation(summary = "更新分割/分级结果", description = "管理员和医生可以更新分割/分级结果")
    @PutMapping("/gradings/{gradingId}")
    public ApiResponse<PlacentaSegmentationGrading> updateGrading(
            @Parameter(description = "分割/分级结果ID", required = true) @PathVariable Long gradingId,
            @Parameter(description = "更新后的分割/分级结果实体", required = true) @RequestBody PlacentaSegmentationGrading grading,
            @RequestHeader("Authorization") String token) {
        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token)) {
            return new ApiResponse<>("success", "Successfully fetched segmentation/grading result", imagingService.updateGrading(gradingId, grading));
        } else {
            return new ApiResponse<>("error", "Unauthorized", null);
        }
    }

    @Operation(summary = "删除分割/分级结果", description = "管理员和医生可以删除分割/分级结果")
    @DeleteMapping("/gradings/{gradingId}")
    public ApiResponse<?> deleteGrading(
            @Parameter(description = "分割/分级结果ID", required = true) @PathVariable Long gradingId,
            @RequestHeader("Authorization") String token) {
        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token)) {
            int status = imagingService.deleteGrading(gradingId);
            return new ApiResponse<>(status==0?"success":"failure", status==0?"Successfully deleted grading result":"failed to delete grading result", status);
        } else {
            return new ApiResponse<>("error", "Unauthorized", null);
        }
    }

    @PostMapping(value = "/multi-segment-image", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    @Operation(summary = "多目标图像分割", description = "对一张图像进行多目标分割，返回多个掩膜信息")
    public ResponseEntity<?> multiSegmentImage(
            @Parameter(description = "影像记录ID", required = true) @RequestParam String recordId,
            @Parameter(description = "图像文件", required = true) @RequestPart("file") MultipartFile image,
            @Parameter(description = "目标类型名称列表，例如：[\"placenta\", \"cord\"]", required = true) 
            @RequestParam(required = true) List<String> targets,
            @Parameter(description = "提示类型(point/box/mask)", required = false) @RequestParam(required = false) String promptType,
            @Parameter(
                description = """
                分割提示坐标映射，key为目标类型名称，value为该类型对应的坐标JSON字符串。
                例如：
                {
                    "placenta": [[10,20], [30,40]],  
                    "bladder": [[50,60], [70,80]]
                }
                坐标格式取决于promptType:
                - point: [[x1,y1], [x2,y2], ...]  
                - box: [[x1,y1,x2,y2], ...]  
                - mask: 二值mask的base64编码
                """,
                schema = @Schema(implementation = Object.class, example = "{\"placenta\":[[10,20],[30,40]],\"bladder\":[[50,60],[70,80]]}")
            )
            @RequestParam Map<String, Object> prompts,
            @RequestHeader("Authorization") String token) throws IOException, InterruptedException {
        
        String patientId = imagingRecordRepository.findById(recordId).get().getPatient().getPatientId();
        if (!utilFunctions.isAdmin(token) && !utilFunctions.isDoctor(token) && !utilFunctions.isMatch(token, patientService.getPatient(patientId).getUser().getUserId())) {
            return ResponseEntity.status(401).build();
        }

        // Add image
        Image savedImage = imagingService.addImage(recordId, image);
        if (savedImage == null) {
            return ResponseEntity.status(500).body("Failed to add image");
        }
        Long imageId = savedImage.getImageId();

        // Remove non-prompt parameters from prompts map
        if (prompts != null) {
            prompts.remove("imageId");
            prompts.remove("promptType"); 
            prompts.remove("targets");
        }

        // Perform multi-target segmentation
        Map<String, String> segmentedImagePaths = segmentService.multiSegmentImagePy(
            patientId.toString(),
            recordId,
            savedImage.getImagePath(),
            promptType,
            targets,
            prompts
        );

        // Add masks for each target
        List<Mask> masks = new ArrayList<>();
        for (Map.Entry<String, String> entry : segmentedImagePaths.entrySet()) {
            Mask addedMask = imagingService.addMask(imageId, Paths.get(entry.getValue()), "MODEL");
            if (addedMask != null) {
                masks.add(addedMask);
            }
        }

        if (masks.isEmpty()) {
            return ResponseEntity.status(404).body("No masks were generated");
        }

        maskRepository.saveAllAndFlush(masks);

        // Create response with all mask information
        MultiValueMap<String, Object> response = new LinkedMultiValueMap<>();
        response.add("image", savedImage);
        response.add("masks", masks);

        return ResponseEntity.ok()
                .contentType(MediaType.APPLICATION_JSON)
                .body(response);
    }

    @PostMapping(value = "/multi-segment-exist-image")
    @Operation(summary = "已有图像多目标分割", description = "对数据库中已有的图像进行多目标分割，返回多个掩膜信息")
    public ResponseEntity<?> multiSegmentExistingImage(
            @Parameter(description = "图像ID", required = true) @RequestParam Long imageId,
            @Parameter(description = "目标类型名称列表，例如：[\"placenta\", \"cord\"]", required = true) 
            @RequestParam List<String> targets,
            @Parameter(description = "提示类型(point/box/mask)", required = false) @RequestParam(required = false) String promptType,
            @Parameter(
                description = """
                分割提示坐标映射，key为目标类型名称，value为该类型对应的坐标JSON字符串。
                例如：
                {
                    "placenta": [[10,20], [30,40]],  
                    "bladder": [[50,60], [70,80]]
                }
                坐标格式取决于promptType:
                - point: [[x1,y1], [x2,y2], ...]  
                - box: [[x1,y1,x2,y2], ...]
                - mask: 二值mask的base64编码
                """,
                schema = @Schema(implementation = Object.class, example = "{\"placenta\":[[10,20],[30,40]],\"bladder\":[[50,60],[70,80]]}")
            )
            @RequestParam Map<String, Object> prompts,
            @RequestHeader("Authorization") String token) throws IOException, InterruptedException {
        
        String patientId = imageRepository.findById(imageId).get().getPatient().getPatientId();
        String recordId = imageRepository.findById(imageId).get().getImagingRecord().getRecordId();
        if (!utilFunctions.isAdmin(token) && !utilFunctions.isDoctor(token) && !utilFunctions.isMatch(token, patientService.getPatient(patientId).getUser().getUserId())) {
            return ResponseEntity.status(401).build();
        }

        // Get existing image
        Image existingImage = imagingService.getImage(imageId);
        if (existingImage == null) {
            return ResponseEntity.status(404).body("Image not found");
        }

        // Remove non-prompt parameters from prompts map
        if (prompts != null) {
            prompts.remove("imageId");
            prompts.remove("promptType"); 
            prompts.remove("targets");
        }

        // Perform multi-target segmentation
        Map<String, String> segmentedImagePaths = segmentService.multiSegmentImagePy(
            patientId.toString(),
            recordId,
            existingImage.getImagePath(),
            promptType,
            targets,
            prompts
        );

        // Add masks for each target
        List<Mask> masks = new ArrayList<>();
        for (Map.Entry<String, String> entry : segmentedImagePaths.entrySet()) {
            Mask addedMask = imagingService.addMask(imageId, Paths.get(entry.getValue()), "MODEL");
            if (addedMask != null) {
                masks.add(addedMask);
            }
        }

        if (masks.isEmpty()) {
            return ResponseEntity.status(404).body("No masks were generated");
        }

        maskRepository.saveAllAndFlush(masks);

        // Create response with all mask information
        MultiValueMap<String, Object> response = new LinkedMultiValueMap<>();
        response.add("image", existingImage);
        response.add("masks", masks);

        return ResponseEntity.ok()
                .contentType(MediaType.APPLICATION_JSON)
                .body(response);
    }

    @PostMapping("/{imageId}/multimodal-classify") 
    @Operation(summary = "多模态分类已有图像", description = "对数据库中已有的图像基于多模态信息进行分类。可选择使用病历记录的相关信息或手动输入。")
    public ResponseEntity<?> multiModalClassify(
            @Parameter(description = "图像ID", required = true) 
            @PathVariable Long imageId,
            @Parameter(description = "使用的模型", required = true, 
                    schema = @Schema(allowableValues = {"mlmpas", "mtpas", "vgg16"}))
            @RequestParam String model,
            @Parameter(description = "年龄（可选，0表示<35岁，1表示>=35岁）。若不提供则从最新病历获取", required = false) 
            @RequestParam(required = false) Integer age,
            @Parameter(description = "前置胎盘类型（可选）：0=无, 1=低置, 2=边缘, 3=部分, 4=完全, 5=凶险。若不提供则从最新病历获取", required = false) 
            @RequestParam(required = false) Integer placentaPrevia,
            @Parameter(description = "剖宫产次数（可选）。若不提供则从最新病历获取", required = false) 
            @RequestParam(required = false) Integer cSectionCount,
            @Parameter(description = "是否有流产史（可选）：0-无, 1-有。若不提供则从最新病历获取", required = false) 
            @RequestParam(required = false) Integer hadAbortion) {
        try {
            Patient patient = imageRepository.findById(imageId).get().getPatient();
            if (patient == null) {
                return ResponseEntity.notFound().build();
            }
            String patientId = patient.getPatientId();

            MedicalRecord latestRecord = null;
            // 只有当任意参数未提供时才获取病历记录
            if (age == null || placentaPrevia == null || cSectionCount == null || hadAbortion == null) {
                List<MedicalRecord> records = prInfoService.findMedicalRecordIdsByPatientId(patientId);
                if (records.isEmpty()) {
                    return ResponseEntity.badRequest().body("No medical records found and not all parameters provided");
                }
                latestRecord = records.get(records.size() - 1);
            }

            // 使用提供的参数或从病历记录获取
            int finalAge = age != null ? age : (latestRecord.getAge() != null ? latestRecord.getAge() : 0);
            int finalPlacentaPrevia = placentaPrevia != null ? placentaPrevia : 
                (latestRecord.getPlacentaPrevia() != null ? latestRecord.getPlacentaPrevia().ordinal() : 0);
            int finalCSectionCount = cSectionCount != null ? cSectionCount : 
                (latestRecord.getCSectionCount() != null ? latestRecord.getCSectionCount() : 0);
            int finalHadAbortion = hadAbortion != null ? hadAbortion : 
                ((latestRecord.getMedicalAbortion() > 0 || latestRecord.getSurgicalAbortion() > 0) ? 1 : 0);

            String imagePath = imageRepository.findById(imageId).get().getImagePath();

            // 使用修改后的参数进行分类
            ClassificationResult result = classificationService.multiModalClassify(
                imagePath,
                finalAge,
                finalPlacentaPrevia,
                finalCSectionCount,
                finalHadAbortion,
                model
            );

            PlacentaClassificationResult savedResult = 
                imagingService.addClassification(imageId, result, model.toUpperCase());

            return ResponseEntity.ok(savedResult);

        } catch (Exception e) {
            return ResponseEntity.internalServerError().body(e.getMessage());
        }
    }

    @PostMapping(value = "/multimodal-classify-new", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    @Operation(summary = "多模态分类新图像", description = "对上传的新图像基于多模态信息进行分类。可选择使用病历记录的相关信息或手动输入。")
    public ResponseEntity<?> multiModalClassifyNew(
            @Parameter(description = "图像文件", required = true) 
            @RequestPart("file") MultipartFile image,
            @Parameter(description = "影像记录ID", required = true) 
            @RequestParam String recordId,
            @Parameter(description = "使用的模型", required = true, 
                    schema = @Schema(allowableValues = {"mlmpas", "mtpas", "vgg16"}))
            @RequestParam String model,
            @Parameter(description = "年龄（可选）。若不提供则从最新病历获取", required = false) 
            @RequestParam(required = false) Integer age,
            @Parameter(description = "前置胎盘等级（可选）：0-正常, 1-低置, 2-部分前置, 3-完全前置。若不提供则从最新病历获取", required = false) 
            @RequestParam(required = false) Integer placentaPrevia,
            @Parameter(description = "剖宫产次数（可选）。若不提供则从最新病历获取", required = false) 
            @RequestParam(required = false) Integer cSectionCount,
            @Parameter(description = "是否有流产史（可选）：0-无, 1-有。若不提供则从最新病历获取", required = false) 
            @RequestParam(required = false) Integer hadAbortion,
            @RequestHeader("Authorization") String token) throws IOException {
        
        String patientId = imagingRecordRepository.findById(recordId).get().getPatient().getPatientId();
        if (!utilFunctions.isAdmin(token) && !utilFunctions.isDoctor(token)) {
            return ResponseEntity.status(401).build();
        }

        Image savedImage = imagingService.addImage(recordId, image);
        if (savedImage == null) {
            return ResponseEntity.status(500).body("Failed to add image");
        }

        try {
            Patient patient = patientService.getPatient(patientId);
            if (patient == null) {
                return ResponseEntity.notFound().build();
            }

            MedicalRecord latestRecord = null;
            // 只有当任意参数未提供时才获取病历记录
            if (age == null || placentaPrevia == null || cSectionCount == null || hadAbortion == null) {
                List<MedicalRecord> records = prInfoService.findMedicalRecordIdsByPatientId(patientId);
                if (records.isEmpty()) {
                    return ResponseEntity.badRequest().body("No medical records found and not all parameters provided");
                }
                latestRecord = records.get(records.size() - 1);
            }

            // 使用提供的参数或从病历记录获取
            int finalAge = age != null ? age : (latestRecord.getAge() != null ? latestRecord.getAge() : 0);
            int finalPlacentaPrevia = placentaPrevia != null ? placentaPrevia : 
                (latestRecord.getPlacentaPrevia() != null ? latestRecord.getPlacentaPrevia().ordinal() : 0);
            int finalCSectionCount = cSectionCount != null ? cSectionCount : 
                (latestRecord.getCSectionCount() != null ? latestRecord.getCSectionCount() : 0);
            int finalHadAbortion = hadAbortion != null ? hadAbortion : 
                ((latestRecord.getMedicalAbortion() > 0 || latestRecord.getSurgicalAbortion() > 0) ? 1 : 0);

            // 使用修改后的参数进行分类
            ClassificationResult result = classificationService.multiModalClassify(
                savedImage.getImagePath(),
                finalAge,
                finalPlacentaPrevia,
                finalCSectionCount, 
                finalHadAbortion,
                model
            );

            PlacentaClassificationResult savedResult = 
                imagingService.addClassification(savedImage.getImageId(), result, model.toUpperCase());

            return ResponseEntity.ok(savedResult);

        } catch (Exception e) {
            return ResponseEntity.internalServerError().body(e.getMessage());
        }
    }
}

