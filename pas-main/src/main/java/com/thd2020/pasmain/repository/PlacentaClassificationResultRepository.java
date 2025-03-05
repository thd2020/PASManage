package com.thd2020.pasmain.repository;

import com.thd2020.pasmain.entity.PlacentaClassificationResult;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import java.util.List;

@Repository
public interface PlacentaClassificationResultRepository extends JpaRepository<PlacentaClassificationResult, Long> {
    List<PlacentaClassificationResult> findByPatient_PatientId(Long patientId);
}
