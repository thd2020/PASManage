package com.thd2020.pasmain.repository;

import com.thd2020.pasmain.entity.SurgeryAndBloodTest;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface SurgeryAndBloodTestRepository extends JpaRepository<SurgeryAndBloodTest, Long> {
}
