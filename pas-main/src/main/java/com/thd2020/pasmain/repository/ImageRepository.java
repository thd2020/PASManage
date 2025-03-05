package com.thd2020.pasmain.repository;

import com.thd2020.pasmain.entity.Image;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface ImageRepository extends JpaRepository<Image, Long> {
    List<Image> findByImagingRecord_RecordId(String recordId);

    List<Image> getAllByImageName(String filename);

    Image getImageByImageName(String filename);

    List<Image> findByPatient_PatientId(Long patientId);
}