package com.thd2020.pasmain.controller;

import com.thd2020.pasmain.dto.ApiResponse;
import com.thd2020.pasmain.entity.*;
import com.thd2020.pasmain.repository.ImagingRecordRepository;
import com.thd2020.pasmain.service.ImagingService;
import com.thd2020.pasmain.util.JwtUtil;
import com.thd2020.pasmain.util.UtilFunctions;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.Resource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.List;
import java.util.Optional;

@RestController
@RequestMapping("/api/v1/imaging")
@Tag(name = "Imaging Controller", description = "影像模块的接口")
public class ImagingController {

    @Autowired
    private ImagingService imagingService;

    @Autowired
    private JwtUtil jwtUtil;

    @Autowired
    private UtilFunctions utilFunctions;
    @Autowired
    private ImagingRecordRepository imagingRecordRepository;

    @Operation(summary = "添加影像记录", description = "管理员和医生可以添加影像记录")
    @PostMapping("/records")
    public ApiResponse<ImagingRecord> addImagingRecord(
            @Parameter(description = "影像记录实体", required = true) @RequestBody ImagingRecord imagingRecord,
            @RequestHeader("Authorization") String token) {
        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token)) {
            return imagingService.addImagingRecord(imagingRecord);
        } else {
            return new ApiResponse<>("error", "Unauthorized", null);
        }
    }

    @Operation(summary = "获取影像记录", description = "获取影像记录及其相关图像")
    @GetMapping("/records/{recordId}")
    public ApiResponse<ImagingRecord> getImagingRecord(
            @Parameter(description = "影像记录ID", required = true) @PathVariable String recordId,
            @RequestHeader("Authorization") String token) {
        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token) || utilFunctions.isMatch(token, imagingRecordRepository.getReferenceById(recordId).getPatient().getPatientId())) {
            return imagingService.getImagingRecord(recordId);
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
        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token) || utilFunctions.isMatch(token, imagingRecordRepository.getReferenceById(recordId).getPatient().getPatientId())) {
            return imagingService.updateImagingRecord(recordId, imagingRecord);
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
            return imagingService.deleteImagingRecord(recordId);
        } else {
            return new ApiResponse<>("error", "Unauthorized", null);
        }
    }

    @Operation(summary = "添加图像", description = "管理员和医生可以添加图像")
    @PostMapping(value = "/images", consumes= MediaType.MULTIPART_FORM_DATA_VALUE)
    public ApiResponse<?> addImage(
            @Parameter(description = "要添加进的检测记录id", required = true) @RequestParam String recordId,
            @Parameter(description = "图像实体", required = true) @RequestPart("file") MultipartFile image,
            @RequestHeader("Authorization") String token) {
        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token)) {
            return imagingService.addImage(recordId, image);
        } else {
            return new ApiResponse<>("error", "Unauthorized", null);
        }
    }

    @Operation(summary = "获取图像", description = "获取图像及其文件")
    @GetMapping("/images/{imageId}")
    public ResponseEntity<Resource> getImage(
            @Parameter(description = "图像ID", required = true) @PathVariable Long imageId,
            @RequestHeader("Authorization") String token) throws IOException {
        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token) || utilFunctions.isMatch(token, imageId)) {
            ApiResponse<Image> response = imagingService.getImage(imageId);
            if (response.getData() != null) {
                Image image = response.getData();
                return ResponseEntity.ok()
                        .contentType(MediaType.IMAGE_JPEG)
                        .header(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=\"" + image.getImageName() + "\"")
                        .body(image.getImageResource());
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
            return imagingService.updateImage(imageId, image);
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
            return imagingService.deleteImage(imageId);
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
            @Parameter(description = "掩膜json实体", required = true)  @RequestPart("json_file") MultipartFile segmentationJson,
            @RequestHeader("Authorization") String token) {
        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token)) {
            return imagingService.addMask(imageId, mask, segmentationJson, source);
        } else {
            return new ApiResponse<>("error", "Unauthorized", null);
        }
    }

    @Operation(summary = "获取掩膜", description = "获取掩膜及其文件")
    @GetMapping("/masks/{maskId}")
    public ResponseEntity<Resource> getMask(
            @Parameter(description = "掩膜ID", required = true) @PathVariable Long maskId,
            @RequestHeader("Authorization") String token) throws IOException {
        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token) || utilFunctions.isMatch(token, maskId)) {
            ApiResponse<Mask> response = imagingService.getMask(maskId);
            if (response.getData() != null) {
                Mask mask = response.getData();
                return ResponseEntity.ok()
                        .contentType(MediaType.IMAGE_PNG)
                        .header(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=\"" + mask.getSegmentationMaskPath() + "\"")
                        .body(mask.getMaskResource());
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
            return imagingService.updateMask(maskId, mask);
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
            return imagingService.deleteMask(maskId);
        } else {
            return new ApiResponse<>("error", "Unauthorized", null);
        }
    }

    @Operation(summary = "添加分割/分级结果", description = "管理员和医生可以添加分割/分级结果")
    @PostMapping("/gradings")
    public ApiResponse<PlacentaSegmentationGrading> addGrading(
            @Parameter(description = "图像ID", required = true) @RequestParam Long imageId,
            @Parameter(description = "掩膜ID", required = true) @RequestParam Long maskId,
            @Parameter(description = "分割/分级结果实体", required = true) @RequestBody PlacentaSegmentationGrading grading,
            @RequestHeader("Authorization") String token) {
        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token)) {
            return imagingService.addGrading(imageId, maskId, grading);
        } else {
            return new ApiResponse<>("error", "Unauthorized", null);
        }
    }

    @Operation(summary = "获取分割/分级结果", description = "获取分割/分级结果及其相关图像和掩膜")
    @GetMapping("/gradings/{gradingId}")
    public ApiResponse<PlacentaSegmentationGrading> getGrading(
            @Parameter(description = "分割/分级结果ID", required = true) @PathVariable Long gradingId,
            @RequestHeader("Authorization") String token) throws IOException {
        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token) || utilFunctions.isMatch(token, gradingId)) {
            return imagingService.getGrading(gradingId);
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
            return imagingService.updateGrading(gradingId, grading);
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
            return imagingService.deleteGrading(gradingId);
        } else {
            return new ApiResponse<>("error", "Unauthorized", null);
        }
    }
}

