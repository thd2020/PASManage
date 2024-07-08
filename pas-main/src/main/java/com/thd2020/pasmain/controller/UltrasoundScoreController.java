package com.thd2020.pasmain.controller;

import com.thd2020.pasmain.dto.ApiResponse;
import com.thd2020.pasmain.entity.UltrasoundScore;
import com.thd2020.pasmain.service.UltrasoundScoreService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/v1/ultrasound")
public class UltrasoundScoreController {

    @Autowired
    private UltrasoundScoreService ultrasoundScoreService;

    @PostMapping
    public ApiResponse<UltrasoundScore> addUltrasoundScore(@RequestBody UltrasoundScore ultrasoundScore) {
        UltrasoundScore createdScore = ultrasoundScoreService.addUltrasoundScore(ultrasoundScore);
        return new ApiResponse<>("success", "Ultrasound score added successfully", createdScore);
    }

    @GetMapping("/{score_id}")
    public ApiResponse<UltrasoundScore> getUltrasoundScore(@PathVariable("score_id") int scoreId) {
        UltrasoundScore ultrasoundScore = ultrasoundScoreService.getUltrasoundScoreById(scoreId);
        return new ApiResponse<>("success", "Ultrasound score fetched successfully", ultrasoundScore);
    }

    @PutMapping("/{score_id}")
    public ApiResponse<UltrasoundScore> updateUltrasoundScore(@PathVariable("score_id") int scoreId, @RequestBody UltrasoundScore ultrasoundScore) {
        UltrasoundScore updatedScore = ultrasoundScoreService.updateUltrasoundScore(scoreId, ultrasoundScore);
        return new ApiResponse<>("success", "Ultrasound score updated successfully", updatedScore);
    }

    @DeleteMapping("/{score_id}")
    public ApiResponse<Void> deleteUltrasoundScore(@PathVariable("score_id") int scoreId) {
        ultrasoundScoreService.deleteUltrasoundScore(scoreId);
        return new ApiResponse<>("success", "Ultrasound score deleted successfully", null);
    }
}
