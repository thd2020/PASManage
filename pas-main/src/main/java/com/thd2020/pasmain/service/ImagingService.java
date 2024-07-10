package com.thd2020.pasmain.service;

import com.thd2020.pasmain.dto.ApiResponse;
import com.thd2020.pasmain.entity.*;
import com.thd2020.pasmain.repository.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.core.io.Resource;
import org.springframework.core.io.UrlResource;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@Service
public class ImagingService {

    @Autowired
    private ImagingRecordRepository imagingRecordRepository;

    @Autowired
    private ImageRepository imageRepository;

    @Autowired
    private MaskRepository maskRepository;

    @Autowired
    private PlacentaSegmentationGradingRepository gradingRepository;

    @Autowired
    private PatientRepository patientRepository;

    private final Path rootLocation = Paths.get("upload-dir");

    public ApiResponse<ImagingRecord> addImagingRecord(ImagingRecord imagingRecord) {
        try {
            ImagingRecord savedRecord = imagingRecordRepository.save(imagingRecord);
            return new ApiResponse<>("success", "Imaging record added successfully", savedRecord);
        } catch (Exception e) {
            return new ApiResponse<>("error", "Failed to add imaging record", null);
        }
    }

    public ApiResponse<ImagingRecord> getImagingRecord(String recordId) {
        Optional<ImagingRecord> record = imagingRecordRepository.findById(recordId);
        if (record.isPresent()) {
            // 获取该记录下所有的影像文件
            List<Image> images = imageRepository.findByImagingRecord_RecordId(recordId);
            record.get().setImages(images);
            return new ApiResponse<>("success", "Imaging record fetched successfully", record.get());
        } else {
            return new ApiResponse<>("error", "Imaging record not found", null);
        }
    }

    public ApiResponse<ImagingRecord> updateImagingRecord(String recordId, ImagingRecord updatedRecord) {
        return imagingRecordRepository.findById(recordId).map(record -> {
            record.setTestType(updatedRecord.getTestType());
            record.setTestDate(updatedRecord.getTestDate());
            record.setResultDescription(updatedRecord.getResultDescription());
            ImagingRecord savedRecord = imagingRecordRepository.save(record);
            return new ApiResponse<>("success", "Imaging record updated successfully", savedRecord);
        }).orElseGet(() -> new ApiResponse<>("error", "Imaging record not found", null));
    }

    public ApiResponse<?> deleteImagingRecord(String recordId) {
        return imagingRecordRepository.findById(recordId).map(record -> {
            // 删除关联的Image和Mask
            List<Image> images = imageRepository.findByImagingRecord_RecordId(recordId);
            for (Image image : images) {
                maskRepository.deleteAll(image.getMasks());
                imageRepository.delete(image);
            }
            imagingRecordRepository.delete(record);
            return new ApiResponse<>("success", "Imaging record deleted successfully", null);
        }).orElseGet(() -> new ApiResponse<>("error", "Imaging record not found", null));
    }

    public ApiResponse<Image> addImage(String recordId, MultipartFile file) {
        try {
            Optional<ImagingRecord> record = imagingRecordRepository.findById(recordId);
            if (record.isPresent()) {
                String filename = file.getOriginalFilename();
                Files.copy(file.getInputStream(), this.rootLocation.resolve(filename));
                Image image = new Image();
                image.setImagingRecord(record.get());
                image.setPatient(record.get().getPatient());
                image.setImageName(filename);
                image.setImagePath(this.rootLocation.resolve(filename).toString());
                Image savedImage = imageRepository.save(image);
                return new ApiResponse<>("success", "Image added successfully", savedImage);
            } else {
                return new ApiResponse<>("error", "Imaging record not found", null);
            }
        } catch (IOException e) {
            return new ApiResponse<>("error", "Failed to add image", null);
        }
    }

    public ApiResponse<Image> getImage(Long imageId) {
        Optional<Image> image = imageRepository.findById(imageId);
        if (image.isPresent()) {
            // 获取图片本身
            try {
                Path file = rootLocation.resolve(image.get().getImagePath());
                Resource resource = new UrlResource(file.toUri());
                if (resource.exists() || resource.isReadable()) {
                    image.get().setImageResource(resource);
                    return new ApiResponse<>("success", "Image fetched successfully", image.get());
                } else {
                    return new ApiResponse<>("error", "File not found or not readable", null);
                }
            } catch (Exception e) {
                return new ApiResponse<>("error", "Failed to fetch image", null);
            }
        } else {
            return new ApiResponse<>("error", "Image not found", null);
        }
    }

    public ApiResponse<Image> updateImage(Long imageId, Image updatedImage) {
        return imageRepository.findById(imageId).map(image -> {
            image.setImageName(updatedImage.getImageName());
            image.setImagePath(updatedImage.getImagePath());
            Image savedImage = imageRepository.save(image);
            return new ApiResponse<>("success", "Image updated successfully", savedImage);
        }).orElseGet(() -> new ApiResponse<>("error", "Image not found", null));
    }

    public ApiResponse<?> deleteImage(Long imageId) {
        return imageRepository.findById(imageId).map(image -> {
            maskRepository.deleteAll(image.getMasks());
            imageRepository.delete(image);
            return new ApiResponse<>("success", "Image deleted successfully", null);
        }).orElseGet(() -> new ApiResponse<>("error", "Image not found", null));
    }

    public ApiResponse<Mask> addMask(Long imageId, MultipartFile file, MultipartFile segmentationJson, String source) {
        try {
            Optional<Image> image = imageRepository.findById(imageId);
            if (image.isPresent()) {
                String filename = file.getOriginalFilename();
                assert filename != null;
                Files.copy(file.getInputStream(), this.rootLocation.resolve(filename));
                String jsonname = segmentationJson.getOriginalFilename();
                assert jsonname != null;
                Files.copy(segmentationJson.getInputStream(), this.rootLocation.resolve(jsonname));
                Mask mask = new Mask();
                mask.setImage(image.get());
                mask.setSegmentationMaskPath(this.rootLocation.resolve(filename).toString());
                mask.setSegmentationJsonPath(this.rootLocation.resolve(jsonname).toString());
                mask.setSegmentationSource(Mask.SegmentationSource.valueOf(source));
                Mask savedMask = maskRepository.save(mask);
                return new ApiResponse<>("success", "Mask added successfully", savedMask);
            } else {
                return new ApiResponse<>("error", "Image not found", null);
            }
        } catch (IOException e) {
            return new ApiResponse<>("error", "Failed to add mask", null);
        }
    }

    public ApiResponse<Mask> getMask(Long maskId) {
        Optional<Mask> mask = maskRepository.findById(maskId);
        if (mask.isPresent()) {
            // 获取掩膜图像
            try {
                Path file = rootLocation.resolve(mask.get().getSegmentationMaskPath());
                Resource resource = new UrlResource(file.toUri());
                if (resource.exists() || resource.isReadable()) {
                    mask.get().setMaskResource(resource);
                    return new ApiResponse<>("success", "Mask fetched successfully", mask.get());
                } else {
                    return new ApiResponse<>("error", "File not found or not readable", null);
                }
            } catch (Exception e) {
                return new ApiResponse<>("error", "Failed to fetch mask", null);
            }
        } else {
            return new ApiResponse<>("error", "Mask not found", null);
        }
    }

    public ApiResponse<Mask> updateMask(Long maskId, Mask updatedMask) {
        return maskRepository.findById(maskId).map(mask -> {
            mask.setSegmentationMaskPath(updatedMask.getSegmentationMaskPath());
            mask.setSegmentationJsonPath(updatedMask.getSegmentationJsonPath());
            mask.setSegmentationSource(updatedMask.getSegmentationSource());
            Mask savedMask = maskRepository.save(mask);
            return new ApiResponse<>("success", "Mask updated successfully", savedMask);
        }).orElseGet(() -> new ApiResponse<>("error", "Mask not found", null));
    }

    public ApiResponse<?> deleteMask(Long maskId) {
        return maskRepository.findById(maskId).map(mask -> {
            maskRepository.delete(mask);
            return new ApiResponse<>("success", "Mask deleted successfully", null);
        }).orElseGet(() -> new ApiResponse<>("error", "Mask not found", null));
    }

    public ApiResponse<PlacentaSegmentationGrading> addGrading(Long imageId, Long maskId, PlacentaSegmentationGrading grading) {
        Optional<Image> image = imageRepository.findById(imageId);
        Optional<Mask> mask = maskRepository.findById(maskId);
        if (image.isPresent() && mask.isPresent()) {
            grading.setImage(image.get());
            grading.setMask(mask.get());
            grading.setPatient(image.get().getPatient());
            grading.setTimestamp(LocalDateTime.now());
            PlacentaSegmentationGrading savedGrading = gradingRepository.save(grading);
            return new ApiResponse<>("success", "Grading added successfully", savedGrading);
        } else {
            return new ApiResponse<>("error", "Image or Mask not found", null);
        }
    }

    public ApiResponse<PlacentaSegmentationGrading> getGrading(Long gradingId) {
        Optional<PlacentaSegmentationGrading> grading = gradingRepository.findById(gradingId);
        if (grading.isPresent()) {
            // 获取原图像和掩膜图像
            try {
                Image image = grading.get().getImage();
                Mask mask = grading.get().getMask();

                Path imagePath = rootLocation.resolve(image.getImagePath());
                Resource imageResource = new UrlResource(imagePath.toUri());

                Path maskPath = rootLocation.resolve(mask.getSegmentationMaskPath());
                Resource maskResource = new UrlResource(maskPath.toUri());

                if ((imageResource.exists() && imageResource.isReadable()) && (maskResource.exists() && maskResource.isReadable())) {
                    grading.get().setImageResource(imageResource);
                    grading.get().setMaskResource(maskResource);
                    return new ApiResponse<>("success", "Grading fetched successfully", grading.get());
                } else {
                    return new ApiResponse<>("error", "File not found or not readable", null);
                }
            } catch (Exception e) {
                return new ApiResponse<>("error", "Failed to fetch grading resources", null);
            }
        } else {
            return new ApiResponse<>("error", "Grading not found", null);
        }
    }

    public ApiResponse<PlacentaSegmentationGrading> updateGrading(Long gradingId, PlacentaSegmentationGrading updatedGrading) {
        return gradingRepository.findById(gradingId).map(grading -> {
            grading.setGrade(updatedGrading.getGrade());
            grading.setProbability(updatedGrading.getProbability());
            grading.setOverallGrade(updatedGrading.getOverallGrade());
            PlacentaSegmentationGrading savedGrading = gradingRepository.save(grading);
            return new ApiResponse<>("success", "Grading updated successfully", savedGrading);
        }).orElseGet(() -> new ApiResponse<>("error", "Grading not found", null));
    }

    public ApiResponse<?> deleteGrading(Long gradingId) {
        return gradingRepository.findById(gradingId).map(grading -> {
            gradingRepository.delete(grading);
            return new ApiResponse<>("success", "Grading deleted successfully", null);
        }).orElseGet(() -> new ApiResponse<>("error", "Grading not found", null));
    }
}