package com.thd2020.pasmain.repository;

import com.thd2020.pasmain.entity.MedicalRecord;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface MedicalRecordRepository extends JpaRepository<MedicalRecord, Long> {
    List<Long> findByPatient_PatientId(Long patientId);
}
