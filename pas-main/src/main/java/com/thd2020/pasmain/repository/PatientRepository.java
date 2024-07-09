package com.thd2020.pasmain.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import com.thd2020.pasmain.entity.Patient;
import java.util.List;
import java.util.Optional;

@Repository
public interface PatientRepository extends JpaRepository<Patient, Long> {
    // 根据身份证查找患者
    Patient findByPassId(String passId);

    // 根据用户ID查找患者
    Optional<Patient> findByUser_UserId(Long userId);

    // 根据姓名查找患者
    List<Patient> findByName(String name);

    // 根据性别查找患者
    List<Patient> findByGender(Patient.Gender gender);

    // 根据医生ID查找患者
    List<Patient> findByDoctor_DoctorId(Long doctorId);
}
