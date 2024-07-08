package com.thd2020.pasmain.service;

import com.thd2020.pasmain.dto.ApiResponse;
import com.thd2020.pasmain.entity.Patient;
import com.thd2020.pasmain.repository.PatientRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.Optional;

@Service
public class PatientService {

    @Autowired
    private PatientRepository patientRepository;

    public ApiResponse<Patient> addPatient(Patient patient) {
        patientRepository.save(patient);
        return new ApiResponse<>("success", "Patient added successfully", patient);
    }

    public ApiResponse<Patient> getPatient(Long patientId) {
        Optional<Patient> patient = patientRepository.findById(patientId);
        return patient.map(value -> new ApiResponse<>("success", "Patient fetched successfully", value)).orElseGet(() -> new ApiResponse<>("error", "Patient not found", null));
    }

    public ApiResponse<Patient> updatePatient(Long patientId, Patient updatedPatient) {
        Optional<Patient> existingPatient = patientRepository.findById(patientId);
        if (existingPatient.isPresent()) {
            Patient patient = existingPatient.get();
            patient.setName(updatedPatient.getName());
            patient.setGender(updatedPatient.getGender());
            patient.setBirthDate(updatedPatient.getBirthDate());
            patient.setAddress(updatedPatient.getAddress());
            patientRepository.save(patient);
            return new ApiResponse<>("success", "Patient updated successfully", patient);
        } else {
            return new ApiResponse<>("error", "Patient not found", null);
        }
    }

    public ApiResponse<String> deletePatient(Long patientId) {
        if (patientRepository.existsById(patientId)) {
            patientRepository.deleteById(patientId);
            return new ApiResponse<>("success", "Patient deleted successfully", null);
        } else {
            return new ApiResponse<>("error", "Patient not found", null);
        }
    }
}
