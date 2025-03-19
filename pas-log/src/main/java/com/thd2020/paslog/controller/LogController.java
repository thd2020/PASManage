package com.thd2020.paslog.controller;

import com.thd2020.paslog.entity.LogEntry;
import com.thd2020.paslog.service.LogService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;

import java.time.LocalDateTime;
import java.util.Collections;
import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/v1/logs")
@Tag(name = "日志管理", description = "系统日志记录与查询接口")
public class LogController {

    @Autowired
    private LogService logService;

    @Operation(summary = "记录系统日志", description = "记录常规系统操作日志")
    @PostMapping("/system")
    public LogEntry logSystemEvent(
            @Parameter(description = "用户ID") @RequestParam String userId,
            @Parameter(description = "服务名称") @RequestParam String serviceName,
            @Parameter(description = "操作类型") @RequestParam String action,
            @Parameter(description = "详细信息") @RequestParam String details) {
        return logService.createLog(userId, action, details, LocalDateTime.now());
    }

    @Operation(summary = "查询日志", description = "根据多个条件查询系统日志")
    @GetMapping("/query")
    public List<LogEntry> queryLogs(
            @Parameter(description = "用户ID") @RequestParam(required = false) String userId,
            @Parameter(description = "日志级别") @RequestParam(required = false) LogEntry.LogLevel level,
            @Parameter(description = "服务名称") @RequestParam(required = false) String serviceName,
            @Parameter(description = "开始时间") @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime start,
            @Parameter(description = "结束时间") @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime end) {
        if (level != null && serviceName != null) {
            return logService.findByLevelAndServiceName(level, serviceName);
        } else if (userId != null) {
            return logService.searchLogs(userId, start, end);
        } else if (level == LogEntry.LogLevel.ERROR) {
            return logService.getErrorLogs(start, end);
        } else if (serviceName != null) {
            return logService.getServiceLogs(serviceName);
        } else if (start != null && end != null) {
            return logService.searchLogs(null, start, end);
        } else {
            return logService.searchLogs();
        }
    }

    @Operation(summary = "记录错误日志", description = "记录系统错误信息，包含详细堆栈信息")
    @PostMapping("/error")
    public LogEntry logError(
            @Parameter(description = "用户ID") @RequestParam String userId,
            @Parameter(description = "服务名称") @RequestParam String serviceName,
            @Parameter(description = "错误描述") @RequestParam String errorDetails,
            @Parameter(description = "堆栈信息") @RequestParam String stackTrace) {
        return logService.logError(userId, serviceName, errorDetails, stackTrace);
    }
}