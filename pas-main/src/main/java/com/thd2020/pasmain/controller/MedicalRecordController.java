package com.thd2020.pasmain.controller;

import com.thd2020.pasmain.dto.ApiResponse;
import com.thd2020.pasmain.entity.MedicalRecord;
import com.thd2020.pasmain.service.MedicalRecordService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/v1/medical-records")
public class MedicalRecordController {

    @Autowired
    private MedicalRecordService medicalRecordService;

    @PostMapping
    public ApiResponse<MedicalRecord> addMedicalRecord(@RequestBody MedicalRecord medicalRecord) {
        MedicalRecord createdRecord = medicalRecordService.addMedicalRecord(medicalRecord);
        return new ApiResponse<>("success", "Medical record added successfully", createdRecord);
    }

    @GetMapping("/{record_id}")
    public ApiResponse<MedicalRecord> getMedicalRecord(@PathVariable("record_id") int recordId) {
        MedicalRecord medicalRecord = medicalRecordService.getMedicalRecordById(recordId);
        return new ApiResponse<>("success", "Medical record fetched successfully", medicalRecord);
    }

    @PutMapping("/{record_id}")
    public ApiResponse<MedicalRecord> updateMedicalRecord(@PathVariable("record_id") int recordId, @RequestBody MedicalRecord medicalRecord) {
        MedicalRecord updatedRecord = medicalRecordService.updateMedicalRecord(recordId, medicalRecord);
        return new ApiResponse<>("success", "Medical record updated successfully", updatedRecord);
    }

    @DeleteMapping("/{record_id}")
    public ApiResponse<Void> deleteMedicalRecord(@PathVariable("record_id") int recordId) {
        medicalRecordService.deleteMedicalRecord(recordId);
        return new ApiResponse<>("success", "Medical record deleted successfully", null);
    }
}
