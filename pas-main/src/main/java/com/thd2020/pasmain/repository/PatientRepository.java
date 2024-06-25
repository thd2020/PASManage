package com.thd2020.pasmain.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import com.thd2020.pasmain.entity.Patient;
import java.util.List;

@Repository
public interface PatientRepository extends JpaRepository<Patient, Long> {
    // 根据身份证查找患者
    Patient findByPassId(String passId);

    // 根据用户ID查找患者
    List<Patient> findByUser_UserId(Long userId);

    // 根据姓名查找患者
    List<Patient> findByName(String name);

    // 根据性别查找患者
    List<Patient> findByGender(Patient.Gender gender);
}
