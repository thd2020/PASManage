package com.thd2020.paslog.aop;

import com.thd2020.paslog.entity.LogEntry;
import com.thd2020.paslog.service.LogService;
import org.aspectj.lang.JoinPoint;
import org.aspectj.lang.annotation.AfterReturning;
import org.aspectj.lang.annotation.Aspect;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.time.LocalDateTime;

@Aspect
@Component
public class LoggingAspect {

    @Autowired
    private LogService logService;

    @AfterReturning(pointcut = "execution(* com.thd2020.pasmain.service.*.*(..))", returning = "result")
    public void logAfterMethodExecution(JoinPoint joinPoint, Object result) {
        // Extract method name
        String methodName = joinPoint.getSignature().getName();

        // Extract method arguments
        Object[] args = joinPoint.getArgs();

        // Extract user information (assuming it's available in the arguments or security context)
        String userId = extractUserId(args);

        // Determine action and details based on method name and arguments
        String action = determineAction(methodName);
        String details = determineDetails(methodName, args, result);

        // Log the action
        logService.createLog(userId, action, details, LocalDateTime.now());
    }

    @AfterReturning(pointcut = "execution(* com.thd2020.pasmain.controller.*.*(..))", returning = "result")
    public void logAfterEndpointExecution(JoinPoint joinPoint, Object result) {
        String methodName = joinPoint.getSignature().getName();
        String className = joinPoint.getTarget().getClass().getSimpleName();
        String serviceName = "pas-main";
        
        // Extract method arguments
        Object[] args = joinPoint.getArgs();

        // Extract user information
        String userId = extractUserId(args);

        // Create detailed log entry
        LogEntry logEntry = new LogEntry();
        logEntry.setUserId(userId);
        logEntry.setServiceName(serviceName);
        logEntry.setClassName(className);
        logEntry.setMethodName(methodName);
        logEntry.setAction(methodName);
        logEntry.setTimestamp(LocalDateTime.now());
        logEntry.setLevel(LogEntry.LogLevel.INFO);
        logEntry.setDetails(determineDetails(methodName, args, result));

        // Save log entry
        logService.createLog(userId, logEntry.getAction(), logEntry.getDetails(), logEntry.getTimestamp());
    }

    private String extractUserId(Object[] args) {
        // Implement logic to extract userId from method arguments or security context
        // For example, if userId is passed as the first argument
        if (args.length > 0 && args[0] instanceof String) {
            return (String) args[0];
        }
        return "unknownUser";
    }

    private String determineAction(String methodName) {
        // Implement logic to determine action based on method name
        return methodName;
    }

    private String determineDetails(String methodName, Object[] args, Object result) {
        // Implement logic to determine details based on method name, arguments, and result
        return "Method: " + methodName + ", Args: " + java.util.Arrays.toString(args) + ", Result: " + result;
    }
}