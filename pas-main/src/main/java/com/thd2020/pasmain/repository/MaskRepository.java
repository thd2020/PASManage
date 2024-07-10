package com.thd2020.pasmain.repository;

import com.thd2020.pasmain.entity.Mask;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface MaskRepository extends JpaRepository<Mask, Long> {
}