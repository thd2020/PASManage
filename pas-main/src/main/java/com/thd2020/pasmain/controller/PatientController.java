package com.thd2020.pasmain.controller;

import com.thd2020.pasmain.dto.ApiResponse;
import com.thd2020.pasmain.entity.Patient;
import com.thd2020.pasmain.service.PatientService;
import com.thd2020.pasmain.util.UtilFunctions;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.parameters.RequestBody;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/v1/patients")
public class PatientController {

    @Autowired
    private PatientService patientService;

    @Autowired
    private UtilFunctions utilFunctions;

    @PostMapping
    @Operation(summary = "添加病人信息", description = "允许管理员添加新的病人信息")
    public ApiResponse<Patient> addPatient(
            @Parameter(description = "JWT token for authentication", required = true)
            @RequestHeader("Authorization") String token,
            @RequestBody Patient patient) {
        if (utilFunctions.isAdmin(token)) {
            return patientService.addPatient(patient);
        } else {
            return new ApiResponse<>("failure", "Unauthorized", null);
        }
    }

    @GetMapping("/{patientId}")
    @Operation(summary = "获取病人信息", description = "允许管理员获取病人详细信息")
    public ApiResponse<Patient> getPatient(
            @Parameter(description = "病人ID", required = true) @PathVariable Long patientId,
            @Parameter(description = "JWT token for authentication", required = true) @RequestHeader("Authorization") String token) {
        if (utilFunctions.isAdmin(token)) {
            return patientService.getPatient(patientId);
        } else {
            return new ApiResponse<>("failure", "Unauthorized", null);
        }
    }

    @PutMapping("/{patientId}")
    @Operation(summary = "更新病人信息", description = "允许管理员更新病人信息")
    public ApiResponse<Patient> updatePatient(
            @Parameter(description = "病人ID", required = true) @PathVariable Long patientId,
            @Parameter(description = "JWT token for authentication", required = true) @RequestHeader("Authorization") String token,
            @RequestBody Patient updatedPatient) {
        if (utilFunctions.isAdmin(token)) {
            return patientService.updatePatient(patientId, updatedPatient);
        } else {
            return new ApiResponse<>("failure", "Unauthorized", null);
        }
    }

    @DeleteMapping("/{patientId}")
    @Operation(summary = "删除病人信息", description = "允许管理员删除病人信息")
    public ApiResponse<String> deletePatient(
            @Parameter(description = "病人ID", required = true) @PathVariable Long patientId,
            @Parameter(description = "JWT token for authentication", required = true) @RequestHeader("Authorization") String token) {
        if (utilFunctions.isAdmin(token)) {
            return patientService.deletePatient(patientId);
        } else {
            return new ApiResponse<>("failure", "Unauthorized", null);
        }
    }
}