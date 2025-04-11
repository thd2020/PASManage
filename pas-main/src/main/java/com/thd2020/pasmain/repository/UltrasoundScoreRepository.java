package com.thd2020.pasmain.repository;

import com.thd2020.pasmain.entity.UltrasoundScore;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface UltrasoundScoreRepository extends JpaRepository<UltrasoundScore, Long> {
    List<UltrasoundScore> findByPatient_PatientId(String patientId);
}