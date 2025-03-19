package com.thd2020.paslog.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.thd2020.paslog.entity.LogEntry;
import com.thd2020.paslog.repository.LogRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;

import com.thd2020.paslog.dto.LogEntryDTO;

@Service
public class LoggingListener {
    
    private static final Logger logger = LoggerFactory.getLogger(LoggingListener.class);
    
    @Autowired
    private ObjectMapper objectMapper;

    @Autowired
    private LogRepository logRepository;

    @KafkaListener(topics = "pas-logs", groupId = "pas-log-group")
    public void listen(String message) {
        try {
            LogEntryDTO dto = objectMapper.readValue(message, LogEntryDTO.class);
            LogEntry logEntry = new LogEntry();
            
            // Map DTO fields to entity
            logEntry.setUserId(dto.getUserId());
            logEntry.setTimestamp(dto.getTimestamp());
            logEntry.setLevel(LogEntry.LogLevel.valueOf(dto.getLevel()));
            logEntry.setServiceName(dto.getServiceName());
            logEntry.setClassName(dto.getClassName());
            logEntry.setMethodName(dto.getMethodName());
            logEntry.setDetails(dto.getDetails());
            logEntry.setIpAddress(dto.getIpAddress());
            logEntry.setRequestPath(dto.getRequestPath());
            logEntry.setRequestMethod(dto.getRequestMethod());
            logEntry.setRequestParams(dto.getRequestParams());
            logEntry.setErrorStack(dto.getErrorStack());
            logEntry.setResponseStatus(dto.getResponseStatus());
            
            // Save to database
            logRepository.save(logEntry);
            
        } catch (Exception e) {
            logger.error("Error processing log message: " + message, e);
        }
    }
}