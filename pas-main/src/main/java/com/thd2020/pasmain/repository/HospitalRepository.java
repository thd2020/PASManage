package com.thd2020.pasmain.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import com.thd2020.pasmain.entity.Hospital;
import java.util.List;

@Repository
public interface HospitalRepository extends JpaRepository<Hospital, Long> {
    // 根据名称查找医院
    List<Hospital> findByName(String name);

    // 根据城市查找医院
    List<Hospital> findByCity(String city);

    // 根据省份查找医院
    List<Hospital> findByProvince(String province);
}
