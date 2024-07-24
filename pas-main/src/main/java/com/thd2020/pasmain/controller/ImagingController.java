package com.thd2020.pasmain.controller;

import ai.onnxruntime.OrtException;
import com.thd2020.pasmain.dto.ApiResponse;
import com.thd2020.pasmain.entity.*;
import com.thd2020.pasmain.repository.ImageRepository;
import com.thd2020.pasmain.repository.ImagingRecordRepository;
import com.thd2020.pasmain.repository.MaskRepository;
import com.thd2020.pasmain.repository.PlacentaSegmentationGradingRepository;
import com.thd2020.pasmain.service.ImagingService;
import com.thd2020.pasmain.service.PatientService;
import com.thd2020.pasmain.service.SegmentService;
import com.thd2020.pasmain.service.UserService;
import com.thd2020.pasmain.util.JwtUtil;
import com.thd2020.pasmain.util.UtilFunctions;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
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
import java.nio.file.Paths;
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

    @PostMapping(value = "/segment-image", consumes= MediaType.MULTIPART_FORM_DATA_VALUE)
    @Operation(summary = "图像分割", description = "基于点或框的提示进行图像分割")
    public ResponseEntity<?> segmentImage(
            @Parameter(description = "影像记录ID", required = true) @RequestParam String recordId,
            @Parameter(description = "图像文件", required = true) @RequestPart("file") MultipartFile image,
            @Parameter(description = "分割提示类型", required = true) @RequestParam String hintType,
            @Parameter(description = "提示坐标", required = false) @RequestParam Map<String, Object> hintCoordinates,
            @RequestHeader("Authorization") String token) throws IOException, OrtException, InterruptedException {
        Long patientId = imagingRecordRepository.findById(recordId).get().getPatient().getPatientId();
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

    @Operation(summary = "获取图像", description = "获取图像及其文件")
    @GetMapping("/images/{imageId}")
    public ResponseEntity<MultiValueMap<String, Object>> getImage(
            @Parameter(description = "图像ID", required = true) @PathVariable Long imageId,
            @RequestHeader("Authorization") String token) throws IOException {
        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token) || utilFunctions.isMatch(token, imageRepository.getReferenceById(imageId).getPatient().getUser().getUserId())) {
            Image image = imagingService.getImage(imageId);
            if (image != null) {
                MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
                body.add("image", new FileSystemResource(image.getImagePath()));
                image.setImageResource(null);
                body.add("details", image);

                return ResponseEntity.ok()
                        .contentType(MediaType.MULTIPART_FORM_DATA)
                        .body(body);
            }
        }
        return ResponseEntity.status(401).build();
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

    @Operation(summary = "获取掩膜", description = "获取掩膜及其文件")
    @GetMapping("/masks/{maskId}")
    public ResponseEntity<?> getMask(
            @Parameter(description = "掩膜ID", required = true) @PathVariable Long maskId,
            @RequestHeader("Authorization") String token) throws IOException {
        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token) || utilFunctions.isMatch(token, maskRepository.getReferenceById(maskId).getImage().getPatient().getUser().getUserId())) {
            Mask mask = imagingService.getMask(maskId);
            if (mask != null) {
                MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
                body.add("mask", new FileSystemResource(mask.getSegmentationMaskPath()));
                mask.setMaskResource(null);
                body.add("details", mask);

                return ResponseEntity.ok()
                        .contentType(MediaType.MULTIPART_FORM_DATA)
                        .body(body);
            }
        }
        return ResponseEntity.status(401).build();
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
}

