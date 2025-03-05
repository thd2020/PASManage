package com.thd2020.paslog.repository;

import com.thd2020.paslog.entity.LogEntry;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface LogRepository extends JpaRepository<LogEntry, Long> {
}