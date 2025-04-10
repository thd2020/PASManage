package com.thd2020.pasmain.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import com.thd2020.pasmain.entity.Department;
import java.util.List;

@Repository
public interface DepartmentRepository extends JpaRepository<Department, Long> {
    // 根据科室名称查找科室
    List<Department> findByDepartmentName(String departmentName);

    // 根据医院ID查找科室
    List<Department> findByHospital_HospitalId(String hospitalId);
}
