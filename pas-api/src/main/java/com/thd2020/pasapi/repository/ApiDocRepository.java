package com.thd2020.pasapi.repository;

import com.thd2020.pasapi.entity.ApiDoc;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface ApiDocRepository extends JpaRepository<ApiDoc, Long> {
}
