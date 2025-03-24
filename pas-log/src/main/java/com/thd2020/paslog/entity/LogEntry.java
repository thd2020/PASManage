package com.thd2020.paslog.entity;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;

import java.time.LocalDateTime;

@Entity
@Getter
@Setter
@Table(name = "log_entries")
public class LogEntry {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "user_id")
    private String userId;
    
    @Column(nullable = false)
    private String action;
    
    @Column(columnDefinition = "TEXT")
    private String details;
    
    @Column(nullable = false)
    private LocalDateTime timestamp;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private LogLevel level;

    @Column(name = "service_name")
    private String serviceName;
    
    @Column(name = "class_name")
    private String className;
    
    @Column(name = "method_name")
    private String methodName;
    
    @Column(name = "error_stack", columnDefinition = "TEXT")
    private String errorStack;
    
    @Column(name = "ip_address")
    private String ipAddress;
    
    @Column(name = "request_path")
    private String requestPath;
    
    @Column(name = "request_method")
    private String requestMethod;
    
    @Column(name = "request_params", columnDefinition = "TEXT")
    private String requestParams;
    
    @Column(name = "response_status")
    private String responseStatus;

    public enum LogLevel {
        INFO, WARN, ERROR, DEBUG
    }
}