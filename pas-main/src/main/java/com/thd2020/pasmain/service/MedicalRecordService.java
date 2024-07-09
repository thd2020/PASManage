package com.thd2020.pasmain.service;

import com.thd2020.pasmain.entity.MedicalRecord;
import com.thd2020.pasmain.entity.Patient;
import com.thd2020.pasmain.repository.MedicalRecordRepository;
import com.thd2020.pasmain.repository.PatientRepository;
import com.thd2020.pasmain.util.UtilFunctions;
import org.springframework.beans.BeanUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class MedicalRecordService {

    @Autowired
    private MedicalRecordRepository medicalRecordRepository;

    @Autowired
    private PatientRepository patientRepository;

    @Autowired
    private UtilFunctions utilFunctions;

    public MedicalRecord addMedicalRecord(MedicalRecord medicalRecord) {
        // 验证患者ID
        Optional<Patient> patientOpt = patientRepository.findById(medicalRecord.getPatient().getPatientId());
        if (patientOpt.isEmpty()) {
            throw new IllegalArgumentException("Invalid patient ID");
        }

        return medicalRecordRepository.save(medicalRecord);
    }

    public MedicalRecord getMedicalRecordById(long recordId) {
        return medicalRecordRepository.findById(recordId)
                .orElseThrow(() -> new IllegalArgumentException("Invalid record ID"));
    }

    public MedicalRecord updateMedicalRecord(int recordId, MedicalRecord medicalRecord) {
        MedicalRecord existingRecord = getMedicalRecordById(recordId);

        // 复制非空字段
        BeanUtils.copyProperties(medicalRecord, existingRecord, utilFunctions.getNullPropertyNames(medicalRecord));

        return medicalRecordRepository.save(existingRecord);
    }

    public void deleteMedicalRecord(int recordId) {
        MedicalRecord medicalRecord = getMedicalRecordById(recordId);
        medicalRecordRepository.delete(medicalRecord);
    }

    public List<Long> findRecordIdsByPatientId(Long patientId) {
        return medicalRecordRepository.findByPatient_PatientId(patientId);
    }
}