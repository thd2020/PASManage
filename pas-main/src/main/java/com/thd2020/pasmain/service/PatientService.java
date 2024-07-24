package com.thd2020.pasmain.service;

import com.thd2020.pasmain.dto.ApiResponse;
import com.thd2020.pasmain.entity.Patient;
import com.thd2020.pasmain.entity.User;
import com.thd2020.pasmain.repository.PatientRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class PatientService {

    @Autowired
    private PatientRepository patientRepository;

    @Autowired
    private UserService userService;

    public Optional<Patient> findPatientByUserId(Long userId) {
        return patientRepository.findByUser_UserId(userId);
    }

    public Patient addPatient(Patient patient) {
        if (patient.getUser() == null) {
            User user = new User();
            user.setUsername(patient.getName());
            user.setPassword("Patient123");
            user.setRole(User.Role.PATIENT);
            userService.registerUser(user);
            patient.setUser(user);
        }
        return patientRepository.save(patient);
    }

    public Patient getPatient(Long patientId) {
        return patientRepository.findById(patientId).isPresent()?patientRepository.findById(patientId).get():null;
    }

    public Patient updatePatient(Long patientId, Patient updatedPatient) {
        Optional<Patient> existingPatient = patientRepository.findById(patientId);
        if (existingPatient.isPresent()) {
            Patient patient = existingPatient.get();
            patient.setName(updatedPatient.getName());
            patient.setGender(updatedPatient.getGender());
            patient.setBirthDate(updatedPatient.getBirthDate());
            patient.setAddress(updatedPatient.getAddress());
            return patientRepository.save(patient);
        } else {
            return null;
        }
    }

    public int deletePatient(Long patientId) {
        if (patientRepository.existsById(patientId)) {
            patientRepository.deleteById(patientId);
            return 0;
        } else {
            return -1;
        }
    }

    public List<Patient> findPatientByName(String name) {
        return patientRepository.findByName(name);
    }

    public Patient findPatientByPassId(String passId) {
        return patientRepository.findByPassId(passId);
    }
}
