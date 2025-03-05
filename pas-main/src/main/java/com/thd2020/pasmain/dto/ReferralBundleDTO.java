package com.thd2020.pasmain.dto;

import java.io.Serializable;
import java.util.List;

import com.thd2020.pasmain.entity.Image;
import com.thd2020.pasmain.entity.Mask;
import com.thd2020.pasmain.entity.MedicalRecord;
import com.thd2020.pasmain.entity.Patient;
import com.thd2020.pasmain.entity.PlacentaSegmentationGrading;
import com.thd2020.pasmain.entity.ReferralRequest;
import com.thd2020.pasmain.entity.SurgeryAndBloodTest;
import com.thd2020.pasmain.entity.UltrasoundScore;
import com.thd2020.pasmain.entity.User;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class ReferralBundleDTO implements Serializable {
    private ReferralRequest referral;
    private Patient patient;
    private List<MedicalRecord> medicalRecords;
    private List<UltrasoundScore> ultrasoundScores;
    private List<SurgeryAndBloodTest> surgeryAndBloodTests;
    private List<PlacentaSegmentationGrading> gradings;
    // Getters and Setters
}

