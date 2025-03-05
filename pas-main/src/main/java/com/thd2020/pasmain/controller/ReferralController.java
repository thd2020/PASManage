package com.thd2020.pasmain.controller;

import java.util.List;
import java.util.Optional;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.thd2020.pasmain.dto.ApiResponse;
import com.thd2020.pasmain.entity.ReferralRequest;
import com.thd2020.pasmain.service.ReferralService;
import com.thd2020.pasmain.util.UtilFunctions;

import io.swagger.v3.oas.annotations.Operation;

@RestController
@RequestMapping("/api/v1/referrals")
public class ReferralController {

    @Autowired
    private UtilFunctions utilFunctions;    

    @Autowired
    private ReferralService referralService;

    @Value("${app.api-key}")
    private String apiKey;  // Inject API Key from application.properties
    
    @PostMapping
    @Operation(summary = "发起转诊请求", description = "允许医生发起转诊请求")
    public ApiResponse<ReferralRequest> createReferral(
            @RequestHeader("Authorization") String token,
            @RequestBody ReferralRequest referralRequest) throws JsonProcessingException {
        if (utilFunctions.isDoctor(token) || utilFunctions.isAdmin(token)) {
            ReferralRequest createdRequest = referralService.sendReferralRequest(referralRequest, token);
            return new ApiResponse<>("success", "Referral request created successfully", createdRequest);
        } else {
            return new ApiResponse<>("failure", "Unauthorized", null);
        }
    }


    @GetMapping("/{requestId}/status")
    @Operation(summary = "获取转诊请求状态", description = "获取指定转诊请求的当前状态")
    public ApiResponse<ReferralRequest> getReferralStatus(@PathVariable Long requestId) {
        ReferralRequest referralRequest = referralService.getReferralRequestById(requestId).get();
        if (referralRequest == null) {
            new ApiResponse<>("failure", "Referral request not found", null);
        }
        return new ApiResponse<>("success", "Referral status fetched", referralRequest);
    }

    @GetMapping("/status")
    @Operation(summary = "获取所有转诊请求状态", description = "获取所有转诊请求的当前状态")
    public ApiResponse<List<ReferralRequest>> getAllReferralStatus() {
        List<ReferralRequest> ReferralRequests = referralService.getAllReferralRequest();
        return new ApiResponse<>("success", "Referral status fetched", ReferralRequests);
    }

    @PostMapping("/response")
    @Operation(summary = "接收转诊请求的审批结果", description = "接收目标医院的转诊请求审批结果")
    public ApiResponse<ReferralRequest> receiveReferralResponse(
        @RequestBody ReferralRequest referralRequest,
        @RequestHeader("API-Key") String receivedApiKey) {
        if (!apiKey.equals(receivedApiKey)) {
            return new ApiResponse<>("failure", "Invalid API key", null);
        }
        ReferralRequest updatedReferral = referralService.receiveReferralResponse(referralRequest);
        return new ApiResponse<>("success", "Referral response received", updatedReferral);
    }
}

