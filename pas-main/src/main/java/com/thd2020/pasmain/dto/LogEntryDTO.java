package com.thd2020.pasmain.dto;

import java.io.Serializable;
import java.time.LocalDateTime;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class LogEntryDTO implements Serializable {
    private String userId;
    private String action;
    private String details;
    private LocalDateTime timestamp;
    private String level;
    private String serviceName;
    private String className;
    private String methodName;
    private String errorStack;
    private String ipAddress;
    private String requestPath;
    private String requestMethod;
    private String requestParams;
    private String responseStatus;

    // Generate getters and setters
}
