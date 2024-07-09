package com.thd2020.pasmain.dto;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;

import java.util.List;

@Getter
@Setter
@AllArgsConstructor
public class RelatedIdsResponse {
    private Long patientId;
    private List<Long> medicalRecordIds;
    private List<Long> surgeryAndBloodTestIds;
    private List<Long> ultrasoundScoreIds;

    // Constructor, getters and setters...
}