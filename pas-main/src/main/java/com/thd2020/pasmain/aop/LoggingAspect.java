package com.thd2020.pasmain.aop;

import org.aspectj.lang.JoinPoint;
import org.aspectj.lang.annotation.AfterReturning;
import org.aspectj.lang.annotation.Aspect;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Component;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.web.context.request.RequestContextHolder;
import org.springframework.web.context.request.ServletRequestAttributes;

import jakarta.servlet.http.HttpServletRequest;
import java.time.LocalDateTime;
import java.util.Arrays;

import com.thd2020.pasmain.dto.LogEntryDTO;

@Aspect
@Component
public class LoggingAspect {

    @Autowired
    private KafkaTemplate<String, String> kafkaTemplate;

    @Autowired
    private ObjectMapper objectMapper;

    private static final String TOPIC = "pas-logs";

    @AfterReturning(pointcut = "execution(* com.thd2020.pasmain.service.*.*(..))", returning = "result")
    public void logAfterMethodExecution(JoinPoint joinPoint, Object result) {
        String logMessage = createLogMessage(joinPoint, result);
        kafkaTemplate.send(TOPIC, logMessage); // Send log to Kafka
    }

    @AfterReturning(pointcut = "execution(* com.thd2020.pasmain.controller.*.*(..))", returning = "result")
    public void logAfterEndpointExecution(JoinPoint joinPoint, Object result) {
        String logMessage = createLogMessage(joinPoint, result);
        kafkaTemplate.send(TOPIC, logMessage); // Send log to Kafka
    }

    private String createLogMessage(JoinPoint joinPoint, Object result) {
        try {
            LogEntryDTO logEntry = new LogEntryDTO();
            
            // Basic information
            logEntry.setTimestamp(LocalDateTime.now());
            logEntry.setLevel("INFO");
            logEntry.setServiceName("pas-main");
            
            // Method information
            String className = joinPoint.getTarget().getClass().getName();
            String methodName = joinPoint.getSignature().getName();
            logEntry.setClassName(className);
            logEntry.setMethodName(methodName);
            
            // Request information
            logEntry.setRequestParams(Arrays.toString(joinPoint.getArgs()));
            
            // If the joinpoint is in a controller, extract HTTP information
            if (className.contains(".controller.")) {
                ServletRequestAttributes attributes = (ServletRequestAttributes) RequestContextHolder.getRequestAttributes();
                if (attributes != null) {
                    HttpServletRequest request = attributes.getRequest();
                    logEntry.setIpAddress(request.getRemoteAddr());
                    logEntry.setRequestPath(request.getRequestURI());
                    logEntry.setRequestMethod(request.getMethod());
                }
            }
            
            // Result information
            logEntry.setDetails(result != null ? result.toString() : "null");
            
            return objectMapper.writeValueAsString(logEntry);
        } catch (Exception e) {
            return "Error creating log message: " + e.getMessage();
        }
    }
}