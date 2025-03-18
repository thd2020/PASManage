package com.thd2020.paslog.service;

import com.thd2020.paslog.entity.LogEntry;
import com.thd2020.paslog.repository.LogRepository;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;

@Service
@Transactional
public class LogService {

    @Autowired
    private LogRepository logRepository;

    public LogEntry createLog(String userId, String action, String details, LocalDateTime timestamp) {
        LogEntry logEntry = new LogEntry();
        logEntry.setUserId(userId);
        logEntry.setAction(action);
        logEntry.setDetails(details);
        logEntry.setTimestamp(timestamp);
        return logRepository.save(logEntry);
    }

    public LogEntry logError(String userId, String serviceName, String errorDetails, String stackTrace) {
        LogEntry logEntry = new LogEntry();
        logEntry.setUserId(userId);
        logEntry.setAction("ERROR");
        logEntry.setDetails(errorDetails);
        logEntry.setTimestamp(LocalDateTime.now());
        logEntry.setLevel(LogEntry.LogLevel.ERROR);
        logEntry.setServiceName(serviceName);
        logEntry.setErrorStack(stackTrace);
        return logRepository.save(logEntry);
    }

    public List<LogEntry> searchLogs(String userId, LocalDateTime start, LocalDateTime end) {
        return logRepository.findByUserIdAndTimestampBetween(userId, start, end);
    }

    public List<LogEntry> getErrorLogs(LocalDateTime start, LocalDateTime end) {
        return logRepository.findByLevelAndTimestampBetween(LogEntry.LogLevel.ERROR, start, end);
    }

    public List<LogEntry> getServiceLogs(String serviceName) {
        return logRepository.findByServiceName(serviceName);
    }
}