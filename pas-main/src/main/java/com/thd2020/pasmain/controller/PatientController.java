package com.thd2020.pasmain.controller;

import com.thd2020.pasmain.dto.ApiResponse;
import com.thd2020.pasmain.entity.Patient;
import com.thd2020.pasmain.service.PatientService;
import com.thd2020.pasmain.util.UtilFunctions;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

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
            return new ApiResponse<>("success", "Patient added successfully", patientService.addPatient(patient));
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
            Patient existedPatient = patientService.getPatient(patientId);
            return new ApiResponse<>(existedPatient!=null?"success":"failure", existedPatient!=null?"Patient fetched successfully":"No such patient", existedPatient);
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
            Patient patient = patientService.updatePatient(patientId, updatedPatient);
            return new ApiResponse<>(patient!=null?"success":"failure", patient!=null?"Patient updated successfully":"No such patient", patient);
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
            int code = patientService.deletePatient(patientId);
            return new ApiResponse<>(code==0?"success":"failure", code==0?"Patient deleted successfully":"No such patient", null);
        } else {
            return new ApiResponse<>("failure", "Unauthorized", null);
        }
    }

    @GetMapping("/name")
    @Operation(summary = "通过姓名获取病人信息", description = "允许管理员或医生通过姓名获取所有同名病人详细信息")
    public ApiResponse<List<Patient>> getPatientByName(
            @Parameter(description = "病人姓名", required = true) @RequestParam String patientName,
            @Parameter(description = "JWT token for authentication", required = true) @RequestHeader("Authorization") String token) {
        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token)) {
            List<Patient> patients = patientService.findPatientByName(patientName);
            return new ApiResponse<>("success", "Patients fetched successfully", patients);
        } else {
            return new ApiResponse<>("failure", "Unauthorized", null);
        }
    }

    @GetMapping("/passId")
    @Operation(summary = "通过身份证号获取病人信息", description = "允许管理员或医生通过身份证号获取病人详细信息")
    public ApiResponse<Patient> getPatientByPassId(
            @Parameter(description = "病人身份证号码", required = true) @RequestParam String passId,
            @Parameter(description = "JWT token for authentication", required = true) @RequestHeader("Authorization") String token) {
        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token)) {
            Patient patient = patientService.findPatientByPassId(passId);
            return new ApiResponse<>(patient!=null?"success":"failure", patient!=null?"Patient fetched successfully":"No such patient", patient);
        } else {
            return new ApiResponse<>("failure", "Unauthorized", null);
        }
    }
}