package com.thd2020.pasmain.controller;

import com.thd2020.pasmain.dto.ApiResponse;
import com.thd2020.pasmain.entity.SurgeryAndBloodTest;
import com.thd2020.pasmain.service.SurgeryAndBloodTestService;
import com.thd2020.pasmain.util.UtilFunctions;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/v1/surgery")
public class SurgeryAndBloodTestController {

    @Autowired
    private SurgeryAndBloodTestService surgeryAndBloodTestService;

    @Autowired
    private UtilFunctions utilFunctions;

    @PostMapping
    @Operation(summary = "添加手术和血常规记录", description = "允许管理员和医生添加新的手术和血常规记录")
    public ApiResponse<SurgeryAndBloodTest> addSurgeryAndBloodTest(
            @Parameter(description = "JWT token用于身份验证", required = true)
            @RequestHeader("Authorization") String token,
            @RequestBody SurgeryAndBloodTest surgeryAndBloodTest) {

        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token)) {
            SurgeryAndBloodTest createdRecord = surgeryAndBloodTestService.addSurgeryAndBloodTest(surgeryAndBloodTest);
            return new ApiResponse<>("success", "Surgery and blood test record added successfully", createdRecord);
        } else {
            return new ApiResponse<>("failure", "Unauthorized", null);
        }
    }

    @GetMapping("/{record_id}")
    @Operation(summary = "获取手术和血常规记录", description = "允许管理员、医生以及病人本人获取手术和血常规记录")
    public ApiResponse<SurgeryAndBloodTest> getSurgeryAndBloodTest(
            @Parameter(description = "记录ID", required = true) @PathVariable("record_id") int recordId,
            @Parameter(description = "JWT token用于身份验证", required = true) @RequestHeader("Authorization") String token) {

        SurgeryAndBloodTest surgeryAndBloodTest = surgeryAndBloodTestService.getSurgeryAndBloodTestById(recordId);
        Long patientUserId = surgeryAndBloodTest.getPatient().getUser().getUserId();

        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token) || utilFunctions.isMatch(token, patientUserId)) {
            return new ApiResponse<>("success", "Surgery and blood test record fetched successfully", surgeryAndBloodTest);
        } else {
            return new ApiResponse<>("failure", "Unauthorized", null);
        }
    }

    @PutMapping("/{record_id}")
    @Operation(summary = "更新手术和血常规记录", description = "允许管理员、医生以及病人本人更新手术和血常规记录")
    public ApiResponse<SurgeryAndBloodTest> updateSurgeryAndBloodTest(
            @Parameter(description = "记录ID", required = true) @PathVariable("record_id") int recordId,
            @Parameter(description = "JWT token用于身份验证", required = true) @RequestHeader("Authorization") String token,
            @RequestBody SurgeryAndBloodTest surgeryAndBloodTest) {

        SurgeryAndBloodTest existingRecord = surgeryAndBloodTestService.getSurgeryAndBloodTestById(recordId);
        Long patientUserId = existingRecord.getPatient().getUser().getUserId();

        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token) || utilFunctions.isMatch(token, patientUserId)) {
            SurgeryAndBloodTest updatedRecord = surgeryAndBloodTestService.updateSurgeryAndBloodTest(recordId, surgeryAndBloodTest);
            return new ApiResponse<>("success", "Surgery and blood test record updated successfully", updatedRecord);
        } else {
            return new ApiResponse<>("failure", "Unauthorized", null);
        }
    }

    @DeleteMapping("/{record_id}")
    @Operation(summary = "删除手术和血常规记录", description = "允许管理员和医生删除手术和血常规记录")
    public ApiResponse<Void> deleteSurgeryAndBloodTest(
            @Parameter(description = "记录ID", required = true) @PathVariable("record_id") int recordId,
            @Parameter(description = "JWT token用于身份验证", required = true) @RequestHeader("Authorization") String token) {

        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token)) {
            surgeryAndBloodTestService.deleteSurgeryAndBloodTest(recordId);
            return new ApiResponse<>("success", "Surgery and blood test record deleted successfully", null);
        } else {
            return new ApiResponse<>("failure", "Unauthorized", null);
        }
    }
}