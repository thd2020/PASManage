package com.thd2020.paslog.controller;

import com.thd2020.paslog.entity.LogEntry;
import com.thd2020.paslog.service.LogService;

import java.time.LocalDateTime;

import org.springframework.beans.factory.annotation.Autowired;
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
}