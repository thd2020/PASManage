package com.thd2020.pasmain.controller;

import com.thd2020.pasmain.dto.ApiResponse;
import com.thd2020.pasmain.entity.SurgeryAndBloodTest;
import com.thd2020.pasmain.service.SurgeryAndBloodTestService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/v1/surgery")
public class SurgeryAndBloodTestController {

    @Autowired
    private SurgeryAndBloodTestService surgeryAndBloodTestService;

    @PostMapping
    public ApiResponse<SurgeryAndBloodTest> addSurgeryAndBloodTest(@RequestBody SurgeryAndBloodTest surgeryAndBloodTest) {
        SurgeryAndBloodTest createdRecord = surgeryAndBloodTestService.addSurgeryAndBloodTest(surgeryAndBloodTest);
        return new ApiResponse<>("success", "Surgery and blood test record added successfully", createdRecord);
    }

    @GetMapping("/{record_id}")
    public ApiResponse<SurgeryAndBloodTest> getSurgeryAndBloodTest(@PathVariable("record_id") int recordId) {
        SurgeryAndBloodTest surgeryAndBloodTest = surgeryAndBloodTestService.getSurgeryAndBloodTestById(recordId);
        return new ApiResponse<>("success", "Surgery and blood test record fetched successfully", surgeryAndBloodTest);
    }

    @PutMapping("/{record_id}")
    public ApiResponse<SurgeryAndBloodTest> updateSurgeryAndBloodTest(@PathVariable("record_id") int recordId, @RequestBody SurgeryAndBloodTest surgeryAndBloodTest) {
        SurgeryAndBloodTest updatedRecord = surgeryAndBloodTestService.updateSurgeryAndBloodTest(recordId, surgeryAndBloodTest);
        return new ApiResponse<>("success", "Surgery and blood test record updated successfully", updatedRecord);
    }

    @DeleteMapping("/{record_id}")
    public ApiResponse<Void> deleteSurgeryAndBloodTest(@PathVariable("record_id") int recordId) {
        surgeryAndBloodTestService.deleteSurgeryAndBloodTest(recordId);
        return new ApiResponse<>("success", "Surgery and blood test record deleted successfully", null);
    }
}
