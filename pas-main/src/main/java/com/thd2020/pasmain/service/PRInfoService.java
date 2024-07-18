package com.thd2020.pasmain.service;

import com.thd2020.pasmain.entity.MedicalRecord;
import com.thd2020.pasmain.entity.Patient;
import com.thd2020.pasmain.entity.SurgeryAndBloodTest;
import com.thd2020.pasmain.entity.UltrasoundScore;
import com.thd2020.pasmain.repository.MedicalRecordRepository;
import com.thd2020.pasmain.repository.PatientRepository;
import com.thd2020.pasmain.repository.SurgeryAndBloodTestRepository;
import com.thd2020.pasmain.repository.UltrasoundScoreRepository;
import com.thd2020.pasmain.util.UtilFunctions;
import org.springframework.beans.BeanUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class PRInfoService {

    @Autowired
    private MedicalRecordRepository medicalRecordRepository;

    @Autowired
    private PatientRepository patientRepository;

    @Autowired
    private SurgeryAndBloodTestRepository surgeryAndBloodTestRepository;

    @Autowired
    private UltrasoundScoreRepository ultrasoundScoreRepository;

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

    public MedicalRecord updateMedicalRecord(long recordId, MedicalRecord medicalRecord) {
        MedicalRecord existingRecord = getMedicalRecordById(recordId);

        // 复制非空字段
        BeanUtils.copyProperties(medicalRecord, existingRecord, utilFunctions.getNullPropertyNames(medicalRecord));

        return medicalRecordRepository.save(existingRecord);
    }

    public void deleteMedicalRecord(long recordId) {
        MedicalRecord medicalRecord = getMedicalRecordById(recordId);
        medicalRecordRepository.delete(medicalRecord);
    }

    public List<MedicalRecord> findMedicalRecordIdsByPatientId(Long patientId) {
        return medicalRecordRepository.findByPatient_PatientId(patientId);
    }


    public SurgeryAndBloodTest addSurgeryAndBloodTest(SurgeryAndBloodTest surgeryAndBloodTest) {
        // 验证患者ID
        Optional<Patient> patientOpt = patientRepository.findById(surgeryAndBloodTest.getPatient().getPatientId());
        if (patientOpt.isEmpty()) {
            throw new IllegalArgumentException("Invalid patient ID");
        }

        return surgeryAndBloodTestRepository.save(surgeryAndBloodTest);
    }

    public SurgeryAndBloodTest getSurgeryAndBloodTestById(long recordId) {
        return surgeryAndBloodTestRepository.findById(recordId)
                .orElseThrow(() -> new IllegalArgumentException("Invalid record ID"));
    }

    public SurgeryAndBloodTest updateSurgeryAndBloodTest(long recordId, SurgeryAndBloodTest surgeryAndBloodTest) {
        SurgeryAndBloodTest existingRecord = getSurgeryAndBloodTestById(recordId);

        // 复制非空字段
        BeanUtils.copyProperties(surgeryAndBloodTest, existingRecord, utilFunctions.getNullPropertyNames(surgeryAndBloodTest));

        return surgeryAndBloodTestRepository.save(existingRecord);
    }

    public void deleteSurgeryAndBloodTest(long recordId) {
        SurgeryAndBloodTest surgeryAndBloodTest = getSurgeryAndBloodTestById(recordId);
        surgeryAndBloodTestRepository.delete(surgeryAndBloodTest);
    }

    public List<SurgeryAndBloodTest> findSBRecordIdsByPatientId(Long patientId) {
        return surgeryAndBloodTestRepository.findByPatient_PatientId(patientId);
    }

    public UltrasoundScore addUltrasoundScore(UltrasoundScore ultrasoundScore) {
        // 验证患者ID
        Optional<Patient> patientOpt = patientRepository.findById(ultrasoundScore.getPatient().getPatientId());
        if (patientOpt.isEmpty()) {
            throw new IllegalArgumentException("Invalid patient ID");
        }

        return ultrasoundScoreRepository.save(ultrasoundScore);
    }

    public UltrasoundScore getUltrasoundScoreById(long scoreId) {
        return ultrasoundScoreRepository.findById(scoreId)
                .orElseThrow(() -> new IllegalArgumentException("Invalid score ID"));
    }

    public UltrasoundScore updateUltrasoundScore(Long scoreId, UltrasoundScore ultrasoundScore) {
        UltrasoundScore existingScore = getUltrasoundScoreById(scoreId);

        // 复制非空字段
        BeanUtils.copyProperties(ultrasoundScore, existingScore, utilFunctions.getNullPropertyNames(ultrasoundScore));

        return ultrasoundScoreRepository.save(existingScore);
    }

    public void deleteUltrasoundScore(long scoreId) {
        UltrasoundScore ultrasoundScore = getUltrasoundScoreById(scoreId);
        ultrasoundScoreRepository.delete(ultrasoundScore);
    }

    public List<UltrasoundScore> findScoreIdsByPatientId(Long patientId) {
        return ultrasoundScoreRepository.findByPatient_PatientId(patientId);
    }
}