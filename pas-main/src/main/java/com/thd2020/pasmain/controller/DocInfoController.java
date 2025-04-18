package com.thd2020.pasmain.controller;

import com.thd2020.pasmain.dto.ApiResponse;
import com.thd2020.pasmain.entity.*;
import com.thd2020.pasmain.service.DocInfoService;
import com.thd2020.pasmain.util.UtilFunctions;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Optional;

@RestController
@RequestMapping("/api/v1/info")
public class DocInfoController {

    @Autowired
    private DocInfoService docInfoService;

    @Autowired
    private UtilFunctions utilFunctions;

    // Hospital endpoints
    @PostMapping("/hospitals")
    @Operation(summary = "添加医院信息", description = "仅管理员可以添加医院信息")
    public ApiResponse<?> addHospital(
            @Parameter(description = "JWT token用于身份验证", required = true) @RequestHeader("Authorization") String token,
            @RequestBody Hospital hospital) {
        if (utilFunctions.isAdmin(token)) {
            try {
                Hospital createdHospital = docInfoService.saveHospital(hospital);
                return new ApiResponse<>("success", "Hospital added successfully", createdHospital);
            }
            catch(Exception e) {
                return new ApiResponse<>("failure", "Hospital added failed", e);
            }
        } else {
            return new ApiResponse<>("failure", "Unauthorized", null);
        }
    }

    @GetMapping("/hospitals/{hospitalId}")
    @Operation(summary = "获取医院信息", description = "所有人都能查看医院信息")
    public ApiResponse<Optional<Hospital>> getHospital(
            @Parameter(description = "医院ID", required = true) @PathVariable String hospitalId) {
        Optional<Hospital> hospital = docInfoService.getHospitalById(hospitalId);
        return new ApiResponse<>(hospital.isPresent()?"success":"failure", hospital.isPresent()?"Hospital fetched successfully":"No such hospital", hospital);
    }

    @PutMapping("/hospitals/{hospitalId}")
    @Operation(summary = "更新医院信息", description = "仅管理员和医生可以更新医院信息")
    public ApiResponse<Optional<Hospital>> updateHospital(
            @Parameter(description = "JWT token用于身份验证", required = true) @RequestHeader("Authorization") String token,
            @Parameter(description = "医院ID", required = true) @PathVariable String hospitalId,
            @RequestBody Hospital updatedHospital) {
        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token)) {
            Optional<Hospital> hospital = docInfoService.updateHospital(hospitalId, updatedHospital);
            return new ApiResponse<>(hospital.isPresent()?"success":"failure", hospital.isPresent()?"Hospital updated successfully":"Hospital does not exist", hospital);
        } else {
            return new ApiResponse<>("failure", "Unauthorized", Optional.empty());
        }
    }

    @DeleteMapping("/hospitals/{hospitalId}")
    @Operation(summary = "删除医院信息", description = "仅管理员可以删除医院信息")
    public ApiResponse<Boolean> deleteHospital(
            @Parameter(description = "JWT token用于身份验证", required = true) @RequestHeader("Authorization") String token,
            @Parameter(description = "医院ID", required = true) @PathVariable String hospitalId) {
        if (utilFunctions.isAdmin(token)) {
            boolean result = docInfoService.deleteHospital(hospitalId);
            return new ApiResponse<>(result?"success":"failure", result?"Hospital deleted successfully":"No such hospital", result);
        } else {
            return new ApiResponse<>("failure", "Unauthorized", false);
        }
    }

    // Department endpoints
    @PostMapping("/departments")
    @Operation(summary = "添加科室信息", description = "仅管理员可以添加科室信息")
    public ApiResponse<?> addDepartment(
            @Parameter(description = "JWT token用于身份验证", required = true) @RequestHeader("Authorization") String token,
            @RequestBody Department department) {
        if (utilFunctions.isAdmin(token)) {
            try {
                Department createdDepartment = docInfoService.saveDepartment(department);
                return new ApiResponse<>("success", "Department added successfully", createdDepartment);
            }
            catch (Exception e) {
                return new ApiResponse<>("failure", "department added error", e);
            }
        } else {
            return new ApiResponse<>("failure", "Unauthorized", null);
        }
    }

    @GetMapping("/departments/{departmentId}")
    @Operation(summary = "获取科室信息", description = "所有人都能查看科室信息")
    public ApiResponse<Optional<Department>> getDepartment(
            @Parameter(description = "科室ID", required = true) @PathVariable Long departmentId) {
        Optional<Department> department = docInfoService.getDepartmentById(departmentId);
        return new ApiResponse<>(department.isPresent()?"success":"failure", department.isPresent()?"Department fetched successfully":"No such department", department);
    }

    @PutMapping("/departments/{departmentId}")
    @Operation(summary = "更新科室信息", description = "仅管理员和医生可以更新科室信息")
    public ApiResponse<Optional<Department>> updateDepartment(
            @Parameter(description = "JWT token用于身份验证", required = true) @RequestHeader("Authorization") String token,
            @Parameter(description = "科室ID", required = true) @PathVariable Long departmentId,
            @RequestBody Department updatedDepartment) {
        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token)) {
            Optional<Department> department = docInfoService.updateDepartment(departmentId, updatedDepartment);
            return new ApiResponse<>(department.isPresent()?"success":"failure", department.isPresent()?"Department updated successfully":"No such department", department);
        } else {
            return new ApiResponse<>("failure", "Unauthorized", Optional.empty());
        }
    }

    @DeleteMapping("/departments/{departmentId}")
    @Operation(summary = "删除科室信息", description = "仅管理员可以删除科室信息")
    public ApiResponse<Boolean> deleteDepartment(
            @Parameter(description = "JWT token用于身份验证", required = true) @RequestHeader("Authorization") String token,
            @Parameter(description = "科室ID", required = true) @PathVariable Long departmentId) {
        if (utilFunctions.isAdmin(token)) {
            boolean result = docInfoService.deleteDepartment(departmentId);
            return new ApiResponse<>(result?"success":"failure", result?"Department deleted successfully":"No such department", result);
        } else {
            return new ApiResponse<>("failure", "Unauthorized", false);
        }
    }

    // Doctor endpoints
    @PostMapping("/doctors")
    @Operation(summary = "添加医生信息", description = "仅管理员和医生可以添加医生信息")
    public ApiResponse<?> addDoctor(
            @Parameter(description = "JWT token用于身份验证", required = true) @RequestHeader("Authorization") String token,
            @Parameter(description = "医生类型, 三级医院医生填写T_DOCTOR，基层医院医生填写B_DOCTOR", required = true) @PathVariable String doctorType,
            @RequestBody Doctor doctor) {
        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token)) {
            try {
                Doctor createdDoctor = docInfoService.saveDoctor(doctor, User.Role.valueOf(doctorType));
                return new ApiResponse<>("success", "Doctor added successfully", createdDoctor);
            }
            catch (Exception e) {
                return new ApiResponse<>("failure", "Doctor added error", e);
            }
        } else {
            return new ApiResponse<>("failure", "Unauthorized", null);
        }
    }

    @GetMapping("/doctors/{doctorId}")
    @Operation(summary = "获取医生信息", description = "所有人都能查看医生信息")
    public ApiResponse<Optional<Doctor>> getDoctor(
            @Parameter(description = "医生ID", required = true) @PathVariable Long doctorId) {
        Optional<Doctor> doctor = docInfoService.getDoctorById(doctorId);
        return new ApiResponse<>(doctor.isPresent()?"success":"failure", doctor.isPresent()?"Doctor fetched successfully":"No such doctor", doctor);
    }

    @PutMapping("/doctors/{doctorId}")
    @Operation(summary = "更新医生信息", description = "仅管理员和医生可以更新医生信息")
    public ApiResponse<Optional<Doctor>> updateDoctor(
            @Parameter(description = "JWT token用于身份验证", required = true) @RequestHeader("Authorization") String token,
            @Parameter(description = "医生ID", required = true) @PathVariable Long doctorId,
            @RequestBody Doctor updatedDoctor) {
        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token)) {
            Optional<Doctor> doctor = docInfoService.updateDoctor(doctorId, updatedDoctor);
            return new ApiResponse<>(doctor.isPresent()?"success":"failure", doctor.isPresent()?"Doctor updated successfully":"No such doctor", doctor);
        } else {
            return new ApiResponse<>("failure", "Unauthorized", Optional.empty());
        }
    }

    @DeleteMapping("/doctors/{doctorId}")
    @Operation(summary = "删除医生信息", description = "仅管理员可以删除医生信息")
    public ApiResponse<Boolean> deleteDoctor(
            @Parameter(description = "JWT token用于身份验证", required = true) @RequestHeader("Authorization") String token,
            @Parameter(description = "医生ID", required = true) @PathVariable Long doctorId) {
        if (utilFunctions.isAdmin(token)) {
            boolean result = docInfoService.deleteDoctor(doctorId);
            return new ApiResponse<>(result?"success":"failure", result?"Doctor deleted successfully":"No such doctor", result);
        } else {
            return new ApiResponse<>("failure", "Unauthorized", false);
        }
    }

    // Additional association methods
    @GetMapping("/doctors/department/{departmentId}")
    @Operation(summary = "获取科室中的所有医生", description = "所有人都能查看科室中的所有医生")
    public ApiResponse<List<Doctor>> getDoctorsByDepartment(
            @Parameter(description = "科室ID", required = true) @PathVariable Long departmentId) {
        List<Doctor> doctors = docInfoService.getDoctorsByDepartment(departmentId);
        return new ApiResponse<>("success", "Doctors fetched successfully", doctors);
    }

    @GetMapping("/departments/hospital/{hospitalId}")
    @Operation(summary = "获取医院中的所有科室", description = "所有人都能查看医院中的所有科室")
    public ApiResponse<List<Department>> getDepartmentsByHospital(
            @Parameter(description = "医院ID", required = true) @PathVariable String hospitalId) {
        List<Department> departments = docInfoService.getDepartmentsByHospital(hospitalId);
        return new ApiResponse<>("success", "Departments fetched successfully", departments);
    }

    @GetMapping("/patients/doctors/{doctorId}")
    @Operation(summary = "获取医生中的所有病人", description = "所有人都能查看医院中的所有病人")
    public ApiResponse<List<Patient>> getPatientsByDoctor(
            @Parameter(description = "医生ID", required = true) @PathVariable Long doctorId) {
        List<Patient> patients = docInfoService.getPatientsByDoctor(doctorId);
        return new ApiResponse<>("success", "Doctors fetched successfully", patients);
    }
}