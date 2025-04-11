package com.thd2020.pasmain.repository;

import com.thd2020.pasmain.entity.ImagingRecord;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface ImagingRecordRepository extends JpaRepository<ImagingRecord, String> {
    List<ImagingRecord> findByPatient_PatientId(String patientId);
}