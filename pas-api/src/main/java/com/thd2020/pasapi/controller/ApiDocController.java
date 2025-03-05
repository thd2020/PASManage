package com.thd2020.pasapi.controller;

import com.thd2020.pasapi.dto.ApiResponse;
import com.thd2020.pasapi.entity.ApiDoc;
import com.thd2020.pasapi.service.ApiDocService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Optional;

@RestController
@RequestMapping("/api/v1/docs")
public class ApiDocController {

    @Autowired
    private ApiDocService apiDocService;

    @GetMapping
    @Operation(summary = "Get all API docs", description = "Retrieve a list of all API documentation entries.")
    public ApiResponse<List<ApiDoc>> getAllApiDocs() {
        List<ApiDoc> apiDocs = apiDocService.getAllApiDocs();
        return new ApiResponse<>("success", "API docs fetched successfully", apiDocs);
    }

    @GetMapping("/{id}")
    @Operation(summary = "Get API doc by ID", description = "Retrieve a specific API documentation entry by ID.")
    public ApiResponse<Optional<ApiDoc>> getApiDocById(
            @Parameter(description = "ID of the API doc to retrieve", required = true)
            @PathVariable Long id) {
        Optional<ApiDoc> apiDoc = apiDocService.getApiDocById(id);
        return new ApiResponse<>(apiDoc.isPresent() ? "success" : "failure", apiDoc.isPresent() ? "API doc fetched successfully" : "No such API doc", apiDoc);
    }

    @PostMapping
    @Operation(summary = "Create a new API doc", description = "Create a new API documentation entry.")
    public ApiResponse<ApiDoc> createApiDoc(
            @Parameter(description = "API doc entity to be created", required = true)
            @RequestBody ApiDoc apiDoc) {
        ApiDoc createdApiDoc = apiDocService.createApiDoc(apiDoc);
        return new ApiResponse<>("success", "API doc created successfully", createdApiDoc);
    }

    @PutMapping("/{id}")
    @Operation(summary = "Update an API doc", description = "Update an existing API documentation entry.")
    public ApiResponse<ApiDoc> updateApiDoc(
            @Parameter(description = "ID of the API doc to update", required = true)
            @PathVariable Long id,
            @Parameter(description = "Updated API doc entity", required = true)
            @RequestBody ApiDoc apiDoc) {
        ApiDoc updatedApiDoc = apiDocService.updateApiDoc(id, apiDoc);
        return new ApiResponse<>(updatedApiDoc != null ? "success" : "failure", updatedApiDoc != null ? "API doc updated successfully" : "No such API doc", updatedApiDoc);
    }

    @DeleteMapping("/{id}")
    @Operation(summary = "Delete an API doc", description = "Delete an existing API documentation entry.")
    public ApiResponse<Void> deleteApiDoc(
            @Parameter(description = "ID of the API doc to delete", required = true)
            @PathVariable Long id) {
        apiDocService.deleteApiDoc(id);
        return new ApiResponse<>("success", "API doc deleted successfully", null);
    }
}