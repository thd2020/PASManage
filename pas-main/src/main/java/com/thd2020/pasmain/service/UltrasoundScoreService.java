package com.thd2020.pasmain.service;

import com.thd2020.pasmain.entity.Patient;
import com.thd2020.pasmain.entity.UltrasoundScore;
import com.thd2020.pasmain.repository.PatientRepository;
import com.thd2020.pasmain.repository.UltrasoundScoreRepository;
import com.thd2020.pasmain.util.utilFunctions;
import org.springframework.beans.BeanUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import java.util.Optional;

@Service
public class UltrasoundScoreService {

    @Autowired
    private UltrasoundScoreRepository ultrasoundScoreRepository;

    @Autowired
    private PatientRepository patientRepository;

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

    public UltrasoundScore updateUltrasoundScore(int scoreId, UltrasoundScore ultrasoundScore) {
        UltrasoundScore existingScore = getUltrasoundScoreById(scoreId);

        // 复制非空字段
        BeanUtils.copyProperties(ultrasoundScore, existingScore, utilFunctions.getNullPropertyNames(ultrasoundScore));

        return ultrasoundScoreRepository.save(existingScore);
    }

    public void deleteUltrasoundScore(int scoreId) {
        UltrasoundScore ultrasoundScore = getUltrasoundScoreById(scoreId);
        ultrasoundScoreRepository.delete(ultrasoundScore);
    }
}
