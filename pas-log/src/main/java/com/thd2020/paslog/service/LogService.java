package com.thd2020.paslog.service;

import com.thd2020.paslog.entity.LogEntry;
import com.thd2020.paslog.repository.LogRepository;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;

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
}