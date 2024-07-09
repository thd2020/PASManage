package com.thd2020.pasmain.controller;

import com.thd2020.pasmain.dto.ApiResponse;
import com.thd2020.pasmain.entity.UltrasoundScore;
import com.thd2020.pasmain.service.UltrasoundScoreService;
import com.thd2020.pasmain.util.UtilFunctions;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/v1/ultrasound")
public class UltrasoundScoreController {

    @Autowired
    private UltrasoundScoreService ultrasoundScoreService;

    @Autowired
    private UtilFunctions utilFunctions;

    @PostMapping
    @Operation(summary = "添加超声评分记录", description = "允许管理员和医生添加新的超声评分记录")
    public ApiResponse<UltrasoundScore> addUltrasoundScore(
            @Parameter(description = "JWT token用于身份验证", required = true)
            @RequestHeader("Authorization") String token,
            @RequestBody UltrasoundScore ultrasoundScore) {

        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token)) {
            UltrasoundScore createdScore = ultrasoundScoreService.addUltrasoundScore(ultrasoundScore);
            return new ApiResponse<>("success", "Ultrasound score added successfully", createdScore);
        } else {
            return new ApiResponse<>("failure", "Unauthorized", null);
        }
    }

    @GetMapping("/{score_id}")
    @Operation(summary = "获取超声评分记录", description = "允许管理员、医生以及病人本人获取超声评分记录")
    public ApiResponse<UltrasoundScore> getUltrasoundScore(
            @Parameter(description = "评分记录ID", required = true) @PathVariable("score_id") int scoreId,
            @Parameter(description = "JWT token用于身份验证", required = true) @RequestHeader("Authorization") String token) {

        UltrasoundScore ultrasoundScore = ultrasoundScoreService.getUltrasoundScoreById(scoreId);
        Long patientUserId = ultrasoundScore.getPatient().getUser().getUserId();

        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token) || utilFunctions.isMatch(token, patientUserId)) {
            return new ApiResponse<>("success", "Ultrasound score fetched successfully", ultrasoundScore);
        } else {
            return new ApiResponse<>("failure", "Unauthorized", null);
        }
    }

    @PutMapping("/{score_id}")
    @Operation(summary = "更新超声评分记录", description = "允许管理员、医生以及病人本人更新超声评分记录")
    public ApiResponse<UltrasoundScore> updateUltrasoundScore(
            @Parameter(description = "评分记录ID", required = true) @PathVariable("score_id") int scoreId,
            @Parameter(description = "JWT token用于身份验证", required = true) @RequestHeader("Authorization") String token,
            @RequestBody UltrasoundScore ultrasoundScore) {

        UltrasoundScore existingScore = ultrasoundScoreService.getUltrasoundScoreById(scoreId);
        Long patientUserId = existingScore.getPatient().getUser().getUserId();

        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token) || utilFunctions.isMatch(token, patientUserId)) {
            UltrasoundScore updatedScore = ultrasoundScoreService.updateUltrasoundScore(scoreId, ultrasoundScore);
            return new ApiResponse<>("success", "Ultrasound score updated successfully", updatedScore);
        } else {
            return new ApiResponse<>("failure", "Unauthorized", null);
        }
    }

    @DeleteMapping("/{score_id}")
    @Operation(summary = "删除超声评分记录", description = "允许管理员和医生删除超声评分记录")
    public ApiResponse<Void> deleteUltrasoundScore(
            @Parameter(description = "评分记录ID", required = true) @PathVariable("score_id") int scoreId,
            @Parameter(description = "JWT token用于身份验证", required = true) @RequestHeader("Authorization") String token) {

        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token)) {
            ultrasoundScoreService.deleteUltrasoundScore(scoreId);
            return new ApiResponse<>("success", "Ultrasound score deleted successfully", null);
        } else {
            return new ApiResponse<>("failure", "Unauthorized", null);
        }
    }
}