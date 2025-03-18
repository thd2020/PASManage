package com.thd2020.paslog.repository;

import com.thd2020.paslog.entity.LogEntry;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;

@Repository
public interface LogRepository extends JpaRepository<LogEntry, Long> {
    List<LogEntry> findByUserIdAndTimestampBetween(String userId, LocalDateTime start, LocalDateTime end);
    List<LogEntry> findByLevelAndTimestampBetween(LogEntry.LogLevel level, LocalDateTime start, LocalDateTime end);
    List<LogEntry> findByServiceName(String serviceName);
    List<LogEntry> findByActionContaining(String action);
    List<LogEntry> findByLevelAndServiceName(LogEntry.LogLevel level, String serviceName);
}