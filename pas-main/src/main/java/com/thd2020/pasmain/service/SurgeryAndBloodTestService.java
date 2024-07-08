package com.thd2020.pasmain.service;

import com.thd2020.pasmain.entity.Patient;
import com.thd2020.pasmain.entity.SurgeryAndBloodTest;
import com.thd2020.pasmain.repository.PatientRepository;
import com.thd2020.pasmain.repository.SurgeryAndBloodTestRepository;
import com.thd2020.pasmain.util.utilFunctions;
import org.springframework.beans.BeanUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import java.util.Optional;

@Service
public class SurgeryAndBloodTestService {

    @Autowired
    private SurgeryAndBloodTestRepository surgeryAndBloodTestRepository;

    @Autowired
    private PatientRepository patientRepository;

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

    public SurgeryAndBloodTest updateSurgeryAndBloodTest(int recordId, SurgeryAndBloodTest surgeryAndBloodTest) {
        SurgeryAndBloodTest existingRecord = getSurgeryAndBloodTestById(recordId);

        // 复制非空字段
        BeanUtils.copyProperties(surgeryAndBloodTest, existingRecord, utilFunctions.getNullPropertyNames(surgeryAndBloodTest));

        return surgeryAndBloodTestRepository.save(existingRecord);
    }

    public void deleteSurgeryAndBloodTest(int recordId) {
        SurgeryAndBloodTest surgeryAndBloodTest = getSurgeryAndBloodTestById(recordId);
        surgeryAndBloodTestRepository.delete(surgeryAndBloodTest);
    }
}
