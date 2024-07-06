package com.thd2020.pasmain.controller;

import com.thd2020.pasmain.dto.ApiResponse;
import com.thd2020.pasmain.entity.Patient;
import com.thd2020.pasmain.service.PatientService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/v1/patients")
public class PatientController {

    @Autowired
    private PatientService patientService;

    @PostMapping
    public ApiResponse<Patient> addPatient(@RequestBody Patient patient) {
        return patientService.addPatient(patient);
    }

    @GetMapping("/{patientId}")
    public ApiResponse<Patient> getPatient(@PathVariable Long patientId) {
        return patientService.getPatient(patientId);
    }

    @PutMapping("/{patientId}")
    public ApiResponse<Patient> updatePatient(@PathVariable Long patientId, @RequestBody Patient updatedPatient) {
        return patientService.updatePatient(patientId, updatedPatient);
    }

    @DeleteMapping("/{patientId}")
    public ApiResponse<String> deletePatient(@PathVariable Long patientId) {
        return patientService.deletePatient(patientId);
    }
}
