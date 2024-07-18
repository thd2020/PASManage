package com.thd2020.pasmain.service;

import com.thd2020.pasmain.entity.*;
import com.thd2020.pasmain.repository.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class DocInfoService {

    @Autowired
    private UserRepository userRepository;

    @Autowired
    private DoctorRepository doctorRepository;

    @Autowired
    private HospitalRepository hospitalRepository;

    @Autowired
    private DepartmentRepository departmentRepository;

    @Autowired
    private PatientRepository patientRepository;

    // Doctor related methods
    public Doctor saveDoctor(Doctor doctor) {
        return doctorRepository.save(doctor);
    }

    public Optional<Doctor> getDoctorById(Long doctorId) {
        return doctorRepository.findById(doctorId);
    }

    public Optional<Doctor> updateDoctor(Long doctorId, Doctor updatedDoctor) {
        return doctorRepository.findById(doctorId).map(doctor -> {
            doctor.setName(updatedDoctor.getName());
            doctor.setTitle(updatedDoctor.getTitle());
            doctor.setDepartment(updatedDoctor.getDepartment());
            return doctorRepository.save(doctor);
        });
    }

    public boolean deleteDoctor(Long doctorId) {
        return doctorRepository.findById(doctorId).map(doctor -> {
            doctorRepository.delete(doctor);
            return true;
        }).orElse(false);
    }

    // Hospital related methods
    public Hospital saveHospital(Hospital hospital) {
        return hospitalRepository.save(hospital);
    }

    public Optional<Hospital> getHospitalById(Long hospitalId) {
        return hospitalRepository.findById(hospitalId);
    }

    public Optional<Hospital> updateHospital(Long hospitalId, Hospital updatedHospital) {
        return hospitalRepository.findById(hospitalId).map(hospital -> {
            hospital.setName(updatedHospital.getName());
            hospital.setAddress(updatedHospital.getAddress());
            hospital.setPhone(updatedHospital.getPhone());
            hospital.setGrade(updatedHospital.getGrade());
            hospital.setProvince(updatedHospital.getProvince());
            hospital.setCity(updatedHospital.getCity());
            hospital.setDistrict(updatedHospital.getDistrict());
            return hospitalRepository.save(hospital);
        });
    }

    public boolean deleteHospital(Long hospitalId) {
        return hospitalRepository.findById(hospitalId).map(hospital -> {
            hospitalRepository.delete(hospital);
            return true;
        }).orElse(false);
    }

    // Department related methods
    public Department saveDepartment(Department department) {
        return departmentRepository.save(department);
    }

    public Optional<Department> getDepartmentById(Long departmentId) {
        return departmentRepository.findById(departmentId);
    }

    public Optional<Department> updateDepartment(Long departmentId, Department updatedDepartment) {
        return departmentRepository.findById(departmentId).map(department -> {
            department.setDepartmentName(updatedDepartment.getDepartmentName());
            department.setPhone(updatedDepartment.getPhone());
            department.setHospital(updatedDepartment.getHospital());
            return departmentRepository.save(department);
        });
    }

    public boolean deleteDepartment(Long departmentId) {
        return departmentRepository.findById(departmentId).map(department -> {
            departmentRepository.delete(department);
            return true;
        }).orElse(false);
    }

    // Additional methods to handle associations
    public List<Doctor> getDoctorsByDepartment(Long departmentId) {
        return doctorRepository.findByDepartment_DepartmentId(departmentId);
    }

    public List<Department> getDepartmentsByHospital(Long hospitalId) {
        return departmentRepository.findByHospital_HospitalId(hospitalId);
    }

    public List<Patient> getPatientsByDoctor(Long doctorId) {
        return patientRepository.findByDoctor_DoctorId(doctorId);
    }

    public List<User> getUsersByHospital(Long hospitalId) {
        return userRepository.findByHospital_HospitalId(hospitalId);
    }
}
