package com.thd2020.pasmain.controller;

import com.thd2020.pasmain.dto.ApiResponse;
import com.thd2020.pasmain.entity.MedicalRecord;
import com.thd2020.pasmain.service.MedicalRecordService;
import com.thd2020.pasmain.util.UtilFunctions;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.parameters.RequestBody;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/v1/medical-records")
public class MedicalRecordController {

    @Autowired
    private MedicalRecordService medicalRecordService;

    @Autowired
    private UtilFunctions utilFunctions;

    @PostMapping
    @Operation(summary = "添加病历记录", description = "允许管理员和医生添加新的病历记录")
    public ApiResponse<MedicalRecord> addMedicalRecord(
            @Parameter(description = "JWT token for authentication", required = true)
            @RequestHeader("Authorization") String token,
            @RequestBody MedicalRecord medicalRecord) {

        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token)) {
            MedicalRecord createdRecord = medicalRecordService.addMedicalRecord(medicalRecord);
            return new ApiResponse<>("success", "Medical record added successfully", createdRecord);
        } else {
            return new ApiResponse<>("failure", "Unauthorized", null);
        }
    }

    @GetMapping("/{record_id}")
    @Operation(summary = "获取病历记录", description = "允许管理员、医生以及病人本人获取病历记录")
    public ApiResponse<MedicalRecord> getMedicalRecord(
            @Parameter(description = "病历记录ID", required = true) @PathVariable("record_id") int recordId,
            @Parameter(description = "JWT token for authentication", required = true) @RequestHeader("Authorization") String token) {

        MedicalRecord medicalRecord = medicalRecordService.getMedicalRecordById(recordId);
        Long patientUserId = medicalRecord.getPatient().getUser().getUserId();

        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token) || utilFunctions.isMatch(token, patientUserId)) {
            return new ApiResponse<>("success", "Medical record fetched successfully", medicalRecord);
        } else {
            return new ApiResponse<>("failure", "Unauthorized", null);
        }
    }

    @PutMapping("/{record_id}")
    @Operation(summary = "更新病历记录", description = "允许管理员、医生以及病人本人更新病历记录")
    public ApiResponse<MedicalRecord> updateMedicalRecord(
            @Parameter(description = "病历记录ID", required = true) @PathVariable("record_id") int recordId,
            @Parameter(description = "JWT token for authentication", required = true) @RequestHeader("Authorization") String token,
            @RequestBody MedicalRecord medicalRecord) {

        MedicalRecord existingRecord = medicalRecordService.getMedicalRecordById(recordId);
        Long patientUserId = existingRecord.getPatient().getUser().getUserId();

        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token) || utilFunctions.isMatch(token, patientUserId)) {
            MedicalRecord updatedRecord = medicalRecordService.updateMedicalRecord(recordId, medicalRecord);
            return new ApiResponse<>("success", "Medical record updated successfully", updatedRecord);
        } else {
            return new ApiResponse<>("failure", "Unauthorized", null);
        }
    }

    @DeleteMapping("/{record_id}")
    @Operation(summary = "删除病历记录", description = "允许管理员和医生删除病历记录")
    public ApiResponse<Void> deleteMedicalRecord(
            @Parameter(description = "病历记录ID", required = true) @PathVariable("record_id") int recordId,
            @Parameter(description = "JWT token for authentication", required = true) @RequestHeader("Authorization") String token) {

        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token)) {
            medicalRecordService.deleteMedicalRecord(recordId);
            return new ApiResponse<>("success", "Medical record deleted successfully", null);
        } else {
            return new ApiResponse<>("failure", "Unauthorized", null);
        }
    }
}