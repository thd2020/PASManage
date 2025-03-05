package com.thd2020.pasmain.repository;

import com.thd2020.pasmain.entity.PlacentaSegmentationGrading;

import java.util.List;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface PlacentaSegmentationGradingRepository extends JpaRepository<PlacentaSegmentationGrading, Long> {

    List<PlacentaSegmentationGrading> findByPatient_PatientId(Long patientId);
    
}