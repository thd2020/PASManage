package com.thd2020.pasmain.controller;

import com.thd2020.pasmain.dto.ApiResponse;
import com.thd2020.pasmain.entity.MedicalRecord;
import com.thd2020.pasmain.entity.SurgeryAndBloodTest;
import com.thd2020.pasmain.entity.UltrasoundScore;
import com.thd2020.pasmain.service.PRInfoService;
import com.thd2020.pasmain.util.UtilFunctions;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.parameters.RequestBody;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/v1/records")
public class PRInfoController {

    @Autowired
    private PRInfoService prInfoService;

    @Autowired
    private UtilFunctions utilFunctions;

    @PostMapping("/med-records")
    @Operation(summary = "添加病历记录", description = "允许管理员和医生添加新的病历记录")
    public ApiResponse<MedicalRecord> addMedicalRecord(
            @Parameter(description = "JWT token for authentication", required = true)
            @RequestHeader("Authorization") String token,
            @org.springframework.web.bind.annotation.RequestBody MedicalRecord medicalRecord) {

        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token)) {
            MedicalRecord createdRecord = prInfoService.addMedicalRecord(medicalRecord);
            return new ApiResponse<>("success", "Medical record added successfully", createdRecord);
        } else {
            return new ApiResponse<>("failure", "Unauthorized", null);
        }
    }

    @GetMapping("/med-records/{record_id}")
    @Operation(summary = "获取病历记录", description = "允许管理员、医生以及病人本人获取病历记录")
    public ApiResponse<MedicalRecord> getMedicalRecord(
            @Parameter(description = "病历记录ID", required = true) @PathVariable("record_id") int recordId,
            @Parameter(description = "JWT token for authentication", required = true) @RequestHeader("Authorization") String token) {

        MedicalRecord medicalRecord = prInfoService.getMedicalRecordById(recordId);
        Long patientUserId = medicalRecord.getPatient().getUser().getUserId();

        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token) || utilFunctions.isMatch(token, patientUserId)) {
            return new ApiResponse<>("success", "Medical record fetched successfully", medicalRecord);
        } else {
            return new ApiResponse<>("failure", "Unauthorized", null);
        }
    }

    @PutMapping("/med-records/{record_id}")
    @Operation(summary = "更新病历记录", description = "允许管理员、医生以及病人本人更新病历记录")
    public ApiResponse<MedicalRecord> updateMedicalRecord(
            @Parameter(description = "病历记录ID", required = true) @PathVariable("record_id") int recordId,
            @Parameter(description = "JWT token for authentication", required = true) @RequestHeader("Authorization") String token,
            @org.springframework.web.bind.annotation.RequestBody MedicalRecord medicalRecord) {

        MedicalRecord existingRecord = prInfoService.getMedicalRecordById(recordId);
        Long patientUserId = existingRecord.getPatient().getUser().getUserId();

        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token) || utilFunctions.isMatch(token, patientUserId)) {
            MedicalRecord updatedRecord = prInfoService.updateMedicalRecord(recordId, medicalRecord);
            return new ApiResponse<>("success", "Medical record updated successfully", updatedRecord);
        } else {
            return new ApiResponse<>("failure", "Unauthorized", null);
        }
    }

    @DeleteMapping("/med-records/{record_id}")
    @Operation(summary = "删除病历记录", description = "允许管理员和医生删除病历记录")
    public ApiResponse<Void> deleteMedicalRecord(
            @Parameter(description = "病历记录ID", required = true) @PathVariable("record_id") int recordId,
            @Parameter(description = "JWT token for authentication", required = true) @RequestHeader("Authorization") String token) {

        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token)) {
            prInfoService.deleteMedicalRecord(recordId);
            return new ApiResponse<>("success", "Medical record deleted successfully", null);
        } else {
            return new ApiResponse<>("failure", "Unauthorized", null);
        }
    }

    @PostMapping("/sb-records")
    @Operation(summary = "添加手术和血常规记录", description = "允许管理员和医生添加新的手术和血常规记录")
    public ApiResponse<SurgeryAndBloodTest> addSurgeryAndBloodTest(
            @Parameter(description = "JWT token用于身份验证", required = true)
            @RequestHeader("Authorization") String token,
            @org.springframework.web.bind.annotation.RequestBody SurgeryAndBloodTest surgeryAndBloodTest) {

        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token)) {
            SurgeryAndBloodTest createdRecord = prInfoService.addSurgeryAndBloodTest(surgeryAndBloodTest);
            return new ApiResponse<>("success", "Surgery and blood test record added successfully", createdRecord);
        } else {
            return new ApiResponse<>("failure", "Unauthorized", null);
        }
    }

    @GetMapping("/sb-records/{record_id}")
    @Operation(summary = "获取手术和血常规记录", description = "允许管理员、医生以及病人本人获取手术和血常规记录")
    public ApiResponse<SurgeryAndBloodTest> getSurgeryAndBloodTest(
            @Parameter(description = "记录ID", required = true) @PathVariable("record_id") int recordId,
            @Parameter(description = "JWT token用于身份验证", required = true) @RequestHeader("Authorization") String token) {

        SurgeryAndBloodTest surgeryAndBloodTest = prInfoService.getSurgeryAndBloodTestById(recordId);
        Long patientUserId = surgeryAndBloodTest.getPatient().getUser().getUserId();

        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token) || utilFunctions.isMatch(token, patientUserId)) {
            return new ApiResponse<>("success", "Surgery and blood test record fetched successfully", surgeryAndBloodTest);
        } else {
            return new ApiResponse<>("failure", "Unauthorized", null);
        }
    }

    @PutMapping("/sb-records/{record_id}")
    @Operation(summary = "更新手术和血常规记录", description = "允许管理员、医生以及病人本人更新手术和血常规记录")
    public ApiResponse<SurgeryAndBloodTest> updateSurgeryAndBloodTest(
            @Parameter(description = "记录ID", required = true) @PathVariable("record_id") int recordId,
            @Parameter(description = "JWT token用于身份验证", required = true) @RequestHeader("Authorization") String token,
            @org.springframework.web.bind.annotation.RequestBody SurgeryAndBloodTest surgeryAndBloodTest) {

        SurgeryAndBloodTest existingRecord = prInfoService.getSurgeryAndBloodTestById(recordId);
        Long patientUserId = existingRecord.getPatient().getUser().getUserId();

        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token) || utilFunctions.isMatch(token, patientUserId)) {
            SurgeryAndBloodTest updatedRecord = prInfoService.updateSurgeryAndBloodTest(recordId, surgeryAndBloodTest);
            return new ApiResponse<>("success", "Surgery and blood test record updated successfully", updatedRecord);
        } else {
            return new ApiResponse<>("failure", "Unauthorized", null);
        }
    }

    @DeleteMapping("/sb-records/{record_id}")
    @Operation(summary = "删除手术和血常规记录", description = "允许管理员和医生删除手术和血常规记录")
    public ApiResponse<Void> deleteSurgeryAndBloodTest(
            @Parameter(description = "记录ID", required = true) @PathVariable("record_id") int recordId,
            @Parameter(description = "JWT token用于身份验证", required = true) @RequestHeader("Authorization") String token) {

        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token)) {
            prInfoService.deleteSurgeryAndBloodTest(recordId);
            return new ApiResponse<>("success", "Surgery and blood test record deleted successfully", null);
        } else {
            return new ApiResponse<>("failure", "Unauthorized", null);
        }
    }

    @PostMapping("/ultra-records")
    @Operation(summary = "添加超声评分记录", description = "允许管理员和医生添加新的超声评分记录")
    public ApiResponse<UltrasoundScore> addUltrasoundScore(
            @Parameter(description = "JWT token用于身份验证", required = true)
            @RequestHeader("Authorization") String token,
            @org.springframework.web.bind.annotation.RequestBody UltrasoundScore ultrasoundScore) {

        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token)) {
            UltrasoundScore createdScore = prInfoService.addUltrasoundScore(ultrasoundScore);
            return new ApiResponse<>("success", "Ultrasound score added successfully", createdScore);
        } else {
            return new ApiResponse<>("failure", "Unauthorized", null);
        }
    }

    @GetMapping("/ultra-records/{score_id}")
    @Operation(summary = "获取超声评分记录", description = "允许管理员、医生以及病人本人获取超声评分记录")
    public ApiResponse<UltrasoundScore> getUltrasoundScore(
            @Parameter(description = "评分记录ID", required = true) @PathVariable("score_id") int scoreId,
            @Parameter(description = "JWT token用于身份验证", required = true) @RequestHeader("Authorization") String token) {

        UltrasoundScore ultrasoundScore = prInfoService.getUltrasoundScoreById(scoreId);
        Long patientUserId = ultrasoundScore.getPatient().getUser().getUserId();

        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token) || utilFunctions.isMatch(token, patientUserId)) {
            return new ApiResponse<>("success", "Ultrasound score fetched successfully", ultrasoundScore);
        } else {
            return new ApiResponse<>("failure", "Unauthorized", null);
        }
    }

    @PutMapping("/ultra-records/{score_id}")
    @Operation(summary = "更新超声评分记录", description = "允许管理员、医生以及病人本人更新超声评分记录")
    public ApiResponse<UltrasoundScore> updateUltrasoundScore(
            @Parameter(description = "评分记录ID", required = true) @PathVariable("score_id") Long scoreId,
            @Parameter(description = "JWT token用于身份验证", required = true) @RequestHeader("Authorization") String token,
            @org.springframework.web.bind.annotation.RequestBody UltrasoundScore ultrasoundScore) {
        UltrasoundScore existingScore = prInfoService.getUltrasoundScoreById(scoreId);
        Long patientUserId = existingScore.getPatient().getUser().getUserId();
        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token) || utilFunctions.isMatch(token, patientUserId)) {
            UltrasoundScore updatedScore = prInfoService.updateUltrasoundScore(scoreId, ultrasoundScore);
            return new ApiResponse<>("success", "Ultrasound score updated successfully", updatedScore);
        } else {
            return new ApiResponse<>("failure", "Unauthorized", null);
        }
    }

    @DeleteMapping("/ultra-records/{score_id}")
    @Operation(summary = "删除超声评分记录", description = "允许管理员和医生删除超声评分记录")
    public ApiResponse<Void> deleteUltrasoundScore(
            @Parameter(description = "评分记录ID", required = true) @PathVariable("score_id") int scoreId,
            @Parameter(description = "JWT token用于身份验证", required = true) @RequestHeader("Authorization") String token) {

        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token)) {
            prInfoService.deleteUltrasoundScore(scoreId);
            return new ApiResponse<>("success", "Ultrasound score deleted successfully", null);
        } else {
            return new ApiResponse<>("failure", "Unauthorized", null);
        }
    }
}