package com.thd2020.pasmain.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import com.thd2020.pasmain.entity.Doctor;
import java.util.List;
import java.util.Optional;

@Repository
public interface DoctorRepository extends JpaRepository<Doctor, Long> {
    // 根据用户ID查找医生
    Optional<Doctor> findByUser_UserId(Long userId);

    // 根据科室ID查找医生
    List<Doctor> findByDepartment_DepartmentId(Long departmentId);

    // 根据姓名查找医生
    List<Doctor> findByName(String name);

    // 根据passId查找医生
    Doctor findByPassId(String passId);
}