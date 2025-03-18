package com.thd2020.paslog.controller;

import com.thd2020.paslog.entity.LogEntry;
import com.thd2020.paslog.service.LogService;

import java.time.LocalDateTime;
import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/v1/logs")
public class LogController {

    @Autowired
    private LogService logService;

    @PostMapping
    public LogEntry createLog(@RequestParam String userId, @RequestParam String action, @RequestParam String details) {
        return logService.createLog(userId, action, details, LocalDateTime.now());
    }

    @GetMapping("/search")
    public List<LogEntry> searchLogs(
            @RequestParam String userId,
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime start,
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime end) {
        return logService.searchLogs(userId, start, end);
    }

    @GetMapping("/errors")
    public List<LogEntry> getErrorLogs(
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime start,
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime end) {
        return logService.getErrorLogs(start, end);
    }

    @GetMapping("/service/{serviceName}")
    public List<LogEntry> getServiceLogs(@PathVariable String serviceName) {
        return logService.getServiceLogs(serviceName);
    }

    @PostMapping("/error")
    public LogEntry logError(
            @RequestParam String userId,
            @RequestParam String serviceName,
            @RequestParam String errorDetails,
            @RequestParam String stackTrace) {
        return logService.logError(userId, serviceName, errorDetails, stackTrace);
    }
}