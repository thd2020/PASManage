package com.thd2020.pasmain.dto;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.util.List;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
public class RelatedIdsResponse {
    private String patientId;
    private List<Long> medicalRecordIds;
    private List<Long> surgeryAndBloodTestIds;
    private List<Long> ultrasoundScoreIds;
    private List<Long> imagingRecordIds;
}