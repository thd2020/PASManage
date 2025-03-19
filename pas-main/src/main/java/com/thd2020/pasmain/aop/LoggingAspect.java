package com.thd2020.pasmain.aop;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.thd2020.pasmain.dto.ApiResponse;
import com.thd2020.pasmain.dto.LogEntryDTO;
import com.thd2020.pasmain.util.JwtUtil;
import org.aspectj.lang.JoinPoint;
import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.AfterReturning;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;
import org.aspectj.lang.reflect.MethodSignature;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Component;
import org.springframework.web.context.request.RequestContextHolder;
import org.springframework.web.context.request.ServletRequestAttributes;
import org.springframework.web.bind.annotation.*;

import jakarta.servlet.http.HttpServletRequest;
import java.lang.reflect.Method;
import java.time.LocalDateTime;
import java.util.Arrays;
import java.util.Optional;
import java.util.stream.Collectors;

@Aspect
@Component
public class LoggingAspect {

    @Autowired
    private KafkaTemplate<String, String> kafkaTemplate;

    @Autowired
    private ObjectMapper objectMapper;

    @Autowired
    private JwtUtil jwtUtil;

    private static final String TOPIC = "pas-logs";

    @Around("execution(* com.thd2020.pasmain.controller.*.*(..))")
    public Object logAroundController(ProceedingJoinPoint joinPoint) throws Throwable {
        LogEntryDTO logEntry = new LogEntryDTO();
        ServletRequestAttributes attributes = (ServletRequestAttributes) RequestContextHolder.getRequestAttributes();
        HttpServletRequest request = attributes != null ? attributes.getRequest() : null;
        
        try {
            // Pre-execution logging setup
            setupBasicLogInfo(logEntry, joinPoint, request);
            
            // Execute the method
            Object result = joinPoint.proceed();
            
            // Post-execution success logging
            handleSuccessResponse(logEntry, result);
            
            return result;
            
        } catch (Exception e) {
            // Handle exception and log error
            handleException(logEntry, e);
            throw e;
        } finally {
            // Always send the log
            sendLog(logEntry);
        }
    }

    private void setupBasicLogInfo(LogEntryDTO logEntry, JoinPoint joinPoint, HttpServletRequest request) {
        logEntry.setTimestamp(LocalDateTime.now());
        logEntry.setLevel("INFO");
        logEntry.setServiceName("pas-main");
        logEntry.setClassName(joinPoint.getTarget().getClass().getName());
        logEntry.setMethodName(joinPoint.getSignature().getName());

        // Extract action from annotation
        Method method = ((MethodSignature) joinPoint.getSignature()).getMethod();
        logEntry.setAction(extractActionFromAnnotations(method));

        if (request != null) {
            logEntry.setIpAddress(getClientIp(request));
            logEntry.setRequestPath(request.getRequestURI());
            logEntry.setRequestMethod(request.getMethod());
            logEntry.setRequestParams(formatRequestParams(joinPoint.getArgs(), request));
            
            // Extract user ID from JWT token
            String token = request.getHeader("Authorization");
            if (token != null && token.startsWith("Bearer ")) {
                try {
                    Long userId = jwtUtil.extractUserId(token.substring(7));
                    logEntry.setUserId(userId.toString());
                } catch (Exception e) {
                    logEntry.setUserId("anonymous");
                }
            } else {
                logEntry.setUserId("anonymous");
            }
        }
    }

    private void handleSuccessResponse(LogEntryDTO logEntry, Object result) {
        logEntry.setLevel("INFO");
        
        if (result instanceof ResponseEntity<?> responseEntity) {
            logEntry.setResponseStatus(responseEntity.getStatusCode().toString());
            logEntry.setDetails(formatResponseDetails(responseEntity.getBody()));
        } else if (result instanceof ApiResponse<?> apiResponse) {
            logEntry.setResponseStatus(apiResponse.getStatus());
            logEntry.setDetails(formatResponseDetails(apiResponse));
        } else {
            logEntry.setResponseStatus("200 OK");
            logEntry.setDetails(formatResponseDetails(result));
        }
    }

    private void handleException(LogEntryDTO logEntry, Exception e) {
        logEntry.setLevel("ERROR");
        logEntry.setResponseStatus("500 Internal Server Error");
        logEntry.setErrorStack(getStackTraceAsString(e));
        logEntry.setDetails("Error: " + e.getMessage());
    }

    private String extractActionFromAnnotations(Method method) {
        // Check for REST annotations and build action string
        StringBuilder action = new StringBuilder();
        
        if (method.isAnnotationPresent(GetMapping.class)) {
            action.append("GET ");
        } else if (method.isAnnotationPresent(PostMapping.class)) {
            action.append("POST ");
        } else if (method.isAnnotationPresent(PutMapping.class)) {
            action.append("PUT ");
        } else if (method.isAnnotationPresent(DeleteMapping.class)) {
            action.append("DELETE ");
        }

        // Add operation summary from Swagger annotation if present
        if (method.isAnnotationPresent(io.swagger.v3.oas.annotations.Operation.class)) {
            action.append(method.getAnnotation(io.swagger.v3.oas.annotations.Operation.class).summary());
        } else {
            action.append(method.getName());
        }
        
        return action.toString();
    }

    private String formatRequestParams(Object[] args, HttpServletRequest request) {
        StringBuilder params = new StringBuilder();
        
        // Add query parameters
        String queryString = request.getQueryString();
        if (queryString != null) {
            params.append("Query: ").append(queryString).append("; ");
        }
        
        // Add method arguments
        if (args != null && args.length > 0) {
            params.append("Body: ").append(
                Arrays.stream(args)
                    .map(this::sanitizeArgument)
                    .collect(Collectors.joining(", "))
            );
        }
        
        return params.toString();
    }

    private String sanitizeArgument(Object arg) {
        if (arg == null) return "null";
        
        // Mask sensitive data in known DTOs
        if (arg.getClass().getSimpleName().contains("Password") || 
            arg.getClass().getSimpleName().contains("Credentials")) {
            return "[MASKED]";
        }
        
        return arg.toString();
    }

    private String formatResponseDetails(Object response) {
        if (response == null) return "null";
        
        try {
            if (response instanceof ApiResponse<?> apiResponse) {
                return String.format("Status: %s, Message: %s, Data: %s",
                    apiResponse.getStatus(),
                    apiResponse.getMessage(),
                    Optional.ofNullable(apiResponse.getData()).map(Object::toString).orElse("null")
                );
            }
            return objectMapper.writeValueAsString(response);
        } catch (Exception e) {
            return response.toString();
        }
    }

    private String getStackTraceAsString(Exception e) {
        StringBuilder sb = new StringBuilder();
        sb.append(e.toString()).append("\n");
        for (StackTraceElement element : e.getStackTrace()) {
            sb.append("\tat ").append(element.toString()).append("\n");
        }
        return sb.toString();
    }

    private String getClientIp(HttpServletRequest request) {
        String clientIp = request.getHeader("X-Forwarded-For");
        if (clientIp == null || clientIp.isEmpty() || "unknown".equalsIgnoreCase(clientIp)) {
            clientIp = request.getHeader("Proxy-Client-IP");
        }
        if (clientIp == null || clientIp.isEmpty() || "unknown".equalsIgnoreCase(clientIp)) {
            clientIp = request.getHeader("WL-Proxy-Client-IP");
        }
        if (clientIp == null || clientIp.isEmpty() || "unknown".equalsIgnoreCase(clientIp)) {
            clientIp = request.getRemoteAddr();
        }
        return clientIp;
    }

    private void sendLog(LogEntryDTO logEntry) {
        try {
            String logMessage = objectMapper.writeValueAsString(logEntry);
            kafkaTemplate.send(TOPIC, logMessage);
        } catch (Exception e) {
            // If we can't send the log, at least print it
            System.err.println("Failed to send log: " + e.getMessage());
        }
    }
}