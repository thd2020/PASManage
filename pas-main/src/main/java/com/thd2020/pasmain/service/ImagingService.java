package com.thd2020.pasmain.service;

import com.thd2020.pasmain.dto.ApiResponse;
import com.thd2020.pasmain.entity.*;
import com.thd2020.pasmain.repository.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.core.io.Resource;
import org.springframework.core.io.UrlResource;

import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
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

    private final Path rootLocation = Paths.get("/", "home",  "thd2020", "pas");

    public List<ImagingRecord> findImagingRecordIds(Long patientId) {
        return imagingRecordRepository.findByPatient_PatientId(patientId);
    }

    public ImagingRecord addImagingRecord(ImagingRecord imagingRecord) {
        return imagingRecordRepository.save(imagingRecord);
    }

    public ImagingRecord getImagingRecord(String recordId) {
        Optional<ImagingRecord> record = imagingRecordRepository.findById(recordId);
        if (record.isPresent()) {
            // 获取该记录下所有的影像文件
            List<Image> images = imageRepository.findByImagingRecord_RecordId(recordId);
            record.get().setImages(images);
            return record.get();
        } else {
            return null;
        }
    }

    public ImagingRecord updateImagingRecord(String recordId, ImagingRecord updatedRecord) {
        return imagingRecordRepository.findById(recordId).map(record -> {
            record.setTestType(updatedRecord.getTestType());
            record.setTestDate(updatedRecord.getTestDate());
            record.setResultDescription(updatedRecord.getResultDescription());
            return imagingRecordRepository.save(record);
        }).orElseGet(() -> null);
    }

    public int deleteImagingRecord(String recordId) {
        return imagingRecordRepository.findById(recordId).map(record -> {
            // 删除关联的Image和Mask
            List<Image> images = imageRepository.findByImagingRecord_RecordId(recordId);
            for (Image image : images) {
                maskRepository.deleteAll(image.getMasks());
                imageRepository.delete(image);
            }
            imagingRecordRepository.delete(record);
            return 0;
        }).orElseGet(() -> -1);
    }

    public Image addImage(String recordId, MultipartFile file) throws IOException {
        Optional<ImagingRecord> record = imagingRecordRepository.findById(recordId);
        if (record.isPresent()) {
            String filename = file.getOriginalFilename();
            Path imageLocation = Paths.get(this.rootLocation.toString(), record.get().getPatient().getPatientId().toString(), recordId, "images");
            Files.createDirectories(imageLocation);
            assert filename != null;
            Files.copy(file.getInputStream(), imageLocation.resolve(filename), StandardCopyOption.REPLACE_EXISTING);
            Image image = new Image();
            image.setImagingRecord(record.get());
            image.setPatient(record.get().getPatient());
            image.setImageName(filename);
            image.setImagePath(imageLocation.resolve(filename).toString());
            return imageRepository.save(image);
        } else {
            return null;
        }
    }

    public Image addImageByPatient(Long patientId, MultipartFile file) throws IOException {
        Optional<Patient> patient = patientRepository.findById(patientId);
        if (patient.isPresent()) {
            String filename = file.getOriginalFilename();
            Path imageLocation = Paths.get(this.rootLocation.toString(), patientId.toString(), "images");
            Files.createDirectories(imageLocation);
            assert filename != null;
            Files.copy(file.getInputStream(), imageLocation.resolve(filename));
            Image image = new Image();
            image.setPatient(patient.get());
            image.setImageName(filename);
            image.setImagePath(this.rootLocation.resolve(filename).toString());
            return imageRepository.save(image);
        } else {
            return null;
        }
    }

    public Image getImage(Long imageId) throws MalformedURLException {
        Optional<Image> image = imageRepository.findById(imageId);
        if (image.isPresent()) {
            // 获取图片本身
            Path file = rootLocation.resolve(image.get().getImagePath());
            Resource resource = new UrlResource(file.toUri());
            if (resource.exists() || resource.isReadable()) {
                image.get().setImageResource(resource);
                return image.get();
            } else {
                return null;
            }
        }
        else {
            return null;
        }
    }

    public Image updateImage(Long imageId, Image updatedImage) {
        return imageRepository.findById(imageId).map(image -> {
            image.setImageName(updatedImage.getImageName());
            image.setImagePath(updatedImage.getImagePath());
            return imageRepository.save(image);
        }).orElseGet(() -> null);
    }

    public int deleteImage(Long imageId) {
        return imageRepository.findById(imageId).map(image -> {
            maskRepository.deleteAll(image.getMasks());
            imageRepository.delete(image);
            return 0;
        }).orElseGet(() -> -1);
    }

    public Mask addMask(Long imageId, MultipartFile file, MultipartFile segmentationJson, String source) throws IOException {
        Optional<Image> image = imageRepository.findById(imageId);
        if (image.isPresent()) {
            String filename = file.getOriginalFilename();
            assert filename != null;
            String recordId = image.get().getImagingRecord().getRecordId();
            Long patientId = image.get().getPatient().getPatientId();
            Path maskLocation = Paths.get(this.rootLocation.toString(), patientId.toString(), recordId, "masks");
            Files.createDirectories(maskLocation);
            Files.copy(file.getInputStream(), maskLocation.resolve(filename));
            String jsonName = segmentationJson.getOriginalFilename();
            assert jsonName != null;
            Files.copy(segmentationJson.getInputStream(), maskLocation.resolve(jsonName));
            Mask mask = new Mask();
            mask.setImage(image.get());
            mask.setSegmentationMaskPath(maskLocation.resolve(filename).toString());
            mask.setSegmentationJsonPath(maskLocation.resolve(jsonName).toString());
            mask.setSegmentationSource(Mask.SegmentationSource.valueOf(source));
            return maskRepository.save(mask);
        } else {
            return null;
        }
    }

    public Mask addMask(Long imageId, MultipartFile file, String source) throws IOException {
        Optional<Image> image = imageRepository.findById(imageId);
        if (image.isPresent()) {
            String filename = file.getOriginalFilename();
            assert filename != null;
            String recordId = image.get().getImagingRecord().getRecordId();
            Long patientId = image.get().getPatient().getPatientId();
            Path maskLocation = Paths.get(this.rootLocation.toString(), patientId.toString(), recordId, "masks");
            Files.createDirectories(maskLocation);
            Files.copy(file.getInputStream(), maskLocation.resolve(filename));
            Mask mask = new Mask();
            mask.setImage(image.get());
            mask.setSegmentationMaskPath(maskLocation.resolve(filename).toString());
            mask.setSegmentationSource(Mask.SegmentationSource.valueOf(source));
            return maskRepository.save(mask);
        } else {
            return null;
        }
    }

    public Mask addMask(Long imageId, Path maskLocation, String source) throws IOException {
        Optional<Image> image = imageRepository.findById(imageId);
        if (image.isPresent()) {
            String recordId = image.get().getImagingRecord().getRecordId();
            Long patientId = image.get().getPatient().getPatientId();
            Files.createDirectories(maskLocation.getParent());
            Mask mask = new Mask();
            mask.setImage(image.get());
            mask.setSegmentationMaskPath(maskLocation.toString());
            mask.setSegmentationSource(Mask.SegmentationSource.valueOf(source));
            return maskRepository.save(mask);
        } else {
            return null;
        }
    }

    public Mask getMask(Long maskId) throws MalformedURLException {
        Optional<Mask> mask = maskRepository.findById(maskId);
        if (mask.isPresent()) {
            // 获取掩膜图像
            Path file = rootLocation.resolve(mask.get().getSegmentationMaskPath());
            Resource resource = new UrlResource(file.toUri());
            if (resource.exists() || resource.isReadable()) {
                mask.get().setMaskResource(resource);
                return mask.get();
            } else {
                return null;
            }
        } else {
            return null;
        }
    }

    public Mask updateMask(Long maskId, Mask updatedMask) {
        return maskRepository.findById(maskId).map(mask -> {
            mask.setSegmentationMaskPath(updatedMask.getSegmentationMaskPath());
            mask.setSegmentationJsonPath(updatedMask.getSegmentationJsonPath());
            mask.setSegmentationSource(updatedMask.getSegmentationSource());
            return maskRepository.save(mask);
        }).orElseGet(() -> null);
    }

    public int deleteMask(Long maskId) {
        return maskRepository.findById(maskId).map(mask -> {
            maskRepository.delete(mask);
            return 0;
        }).orElseGet(() -> -1);
    }

    public PlacentaSegmentationGrading addGrading(Long imageId, Long maskId, PlacentaSegmentationGrading grading) {
        Optional<Image> image = imageRepository.findById(imageId);
        Optional<Mask> mask = maskRepository.findById(maskId);
        if (image.isPresent() && mask.isPresent()) {
            grading.setImage(image.get());
            grading.setMask(mask.get());
            grading.setPatient(image.get().getPatient());
            grading.setTimestamp(LocalDateTime.now());
            return gradingRepository.save(grading);
        } else {
            return null;
        }
    }

    public PlacentaSegmentationGrading getGrading(Long gradingId) throws MalformedURLException {
        Optional<PlacentaSegmentationGrading> grading = gradingRepository.findById(gradingId);
        if (grading.isPresent()) {
            // 获取原图像和掩膜图像
            Image image = grading.get().getImage();
            Mask mask = grading.get().getMask();

            Path imagePath = rootLocation.resolve(image.getImagePath());
            Resource imageResource = new UrlResource(imagePath.toUri());

            Path maskPath = rootLocation.resolve(mask.getSegmentationMaskPath());
            Resource maskResource = new UrlResource(maskPath.toUri());

            if ((imageResource.exists() && imageResource.isReadable()) && (maskResource.exists() && maskResource.isReadable())) {
                grading.get().setImageResource(imageResource);
                grading.get().setMaskResource(maskResource);
                return grading.get();
            } else {
                return null;
            }
        } else {
            return null;
        }
    }

    public PlacentaSegmentationGrading updateGrading(Long gradingId, PlacentaSegmentationGrading updatedGrading) {
        return gradingRepository.findById(gradingId).map(grading -> {
            grading.setGrade(updatedGrading.getGrade());
            grading.setProbability(updatedGrading.getProbability());
            grading.setOverallGrade(updatedGrading.getOverallGrade());
            return gradingRepository.save(grading);
        }).orElseGet(() -> null);
    }

    public int deleteGrading(Long gradingId) {
        return gradingRepository.findById(gradingId).map(grading -> {
            gradingRepository.delete(grading);
            return 0;
        }).orElseGet(() -> -1);
    }
}