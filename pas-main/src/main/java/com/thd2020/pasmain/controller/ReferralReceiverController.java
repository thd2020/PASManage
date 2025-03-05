package com.thd2020.pasmain.controller;

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

import com.thd2020.pasmain.dto.ApiResponse;
import com.thd2020.pasmain.dto.ReferralBundleDTO;
import com.thd2020.pasmain.entity.ReferralRequest;
import com.thd2020.pasmain.service.ReferralReceiverService;
import com.thd2020.pasmain.util.UtilFunctions;

import io.swagger.v3.oas.annotations.Operation;

@RestController
@RequestMapping("/api/v1/referrals")
public class ReferralReceiverController {

    @Autowired
    private ReferralReceiverService referralReceiverService;

    @Autowired
    private UtilFunctions utilFunctions;

    @Value("${app.api-key}")
    private String apiKey;  // Inject API Key from application.properties

    @PostMapping("/receive")
    @Operation(summary = "接收转诊请求", description = "接收来自另一个医院的转诊请求并保存到数据库")
    public ApiResponse<ReferralRequest> receiveReferralRequest(
            @RequestHeader("API-Key") String receivedApiKey,
            @RequestBody ReferralBundleDTO referralBundle) {
            if (!apiKey.equals(receivedApiKey)) {
                return new ApiResponse<>("failure", "Invalid API key", null);
            }
        ReferralRequest savedRequest = referralReceiverService.receiveReferral(referralBundle);
        return new ApiResponse<>("success", "Referral request received and saved", savedRequest);
    }

    @PutMapping("/{requestId}/approve")
    @Operation(summary = "批准转诊请求", description = "医院批准转诊请求并通知发起医院")
    public ApiResponse<ReferralRequest> approveReferral(
        @PathVariable Long requestId, 
        @RequestParam String approvalReason,
        @RequestHeader("Authorization") String token) {
        if (utilFunctions.isAdmin(token) || utilFunctions.isAdmin(token)) {
            ReferralRequest approvalRequest = referralReceiverService.handleReferralResponse(requestId, ReferralRequest.Status.APPROVED, approvalReason);
            return new ApiResponse<>("success", "Referral request accepted", approvalRequest);
        } else {
            return new ApiResponse<>("failure", "Unauthorized", null);
        }
    } 
        
    @PutMapping("/{requestId}/reject")
    @Operation(summary = "拒绝转诊请求", description = "医院拒绝转诊请求")
    public ApiResponse<ReferralRequest> rejectReferral(
        @RequestHeader("Authorization") String token,
        @PathVariable Long requestId,
        @RequestParam String rejectionReason) {
        if (utilFunctions.isAdmin(token) || utilFunctions.isAdmin(token)) {
            ReferralRequest rejectedRequest = referralReceiverService.handleReferralResponse(requestId, ReferralRequest.Status.REJECTED, rejectionReason);
            return new ApiResponse<>("success", "Referral request rejected", rejectedRequest);
        } else {
            return new ApiResponse<>("failure", "Unauthorized", null);
        }
    } 
}
