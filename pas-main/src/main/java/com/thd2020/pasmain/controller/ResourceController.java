package com.thd2020.pasmain.controller;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import org.springframework.validation.annotation.Validated;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.cache.annotation.Cacheable;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import lombok.extern.slf4j.Slf4j;
import com.thd2020.pasmain.dto.ResourceDTO;
import com.thd2020.pasmain.entity.Resource;
import com.thd2020.pasmain.exception.ResourceNotFoundException;
import com.thd2020.pasmain.service.ResourceFetcherService;
import com.thd2020.pasmain.service.ResourceService;
import com.thd2020.pasmain.util.UtilFunctions;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.FileSystemResource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.server.ResponseStatusException;

import java.io.File;
import java.io.IOException;
import java.util.List;

/**
 * REST Controller for managing resources in the PAS system.
 * Provides endpoints for resource retrieval, download, and management.
 *
 * @author YourName
 * @version 1.0
 * @since 2024-01-01
 */
@Slf4j
@Validated
@RestController
@RequestMapping("/api/v1/resources")
@Tag(name = "Resource Management", description = "APIs for managing and retrieving resources")
@CrossOrigin(origins = "*", maxAge = 3600)
public class ResourceController {
    @Autowired
    private ResourceService resourceService;

    @Autowired
    private ResourceFetcherService resourceFetcherService;

    @Autowired
    private UtilFunctions utilFunctions;

    @Operation(summary = "Get all resources", description = "Retrieves a list of all available resources")
    @ApiResponses(value = {
        @ApiResponse(responseCode = "200", description = "Successfully retrieved resources"),
        @ApiResponse(responseCode = "403", description = "Forbidden"),
        @ApiResponse(responseCode = "500", description = "Internal server error")
    })
    @Cacheable(value = "resourcesList", unless = "#result == null")
    @GetMapping(produces = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<List<ResourceDTO>> getAllResources(
            @Parameter(description = "JWT token for authentication", required = true)
            @RequestHeader("Authorization") String token) {
        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token)) {
            return ResponseEntity.ok(resourceService.getAllResources());
        }
        return ResponseEntity.status(HttpStatus.FORBIDDEN).build();
    }

    @Operation(summary = "Get resources by category")
    @ApiResponses(value = {
        @ApiResponse(responseCode = "200", description = "Successfully retrieved resources"),
        @ApiResponse(responseCode = "404", description = "Category not found")
    })
    @GetMapping(value = "/category/{category}", produces = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<List<ResourceDTO>> getResourcesByCategory(
            @Parameter(description = "Category name", required = true)
            @PathVariable @NotBlank String category) {
        return ResponseEntity.ok(resourceService.getResourcesByCategory(category));
    }

    @Operation(summary = "Get resources by type")
    @GetMapping(value = "/type/{resourceType}", produces = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<List<ResourceDTO>> getResourcesByType(
            @Parameter(description = "Resource type", required = true)
            @PathVariable @NotBlank String resourceType) {
        return ResponseEntity.ok(resourceService.getResourcesByType(resourceType));
    }

    @Operation(summary = "Get latest resources")
    @GetMapping(value = "/latest", produces = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<List<ResourceDTO>> getLatestResources(
            @Parameter(description = "Number of resources to return")
            @RequestParam(defaultValue = "10") @Min(1) int limit) {
        return ResponseEntity.ok(resourceService.getLatestResources(limit));
    }

    @Operation(summary = "Trigger manual resource fetch")
    @PostMapping("/fetch")
    public ResponseEntity<Void> triggerManualFetch(
            @Parameter(description = "JWT token for authentication", required = true)
            @RequestHeader("Authorization") String token) {
        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token)) {
            resourceService.triggerManualFetch();
            return ResponseEntity.accepted().build();
        }
        return ResponseEntity.status(HttpStatus.FORBIDDEN).build();
    }

    @Operation(summary = "Get resource by ID")
    @ApiResponses(value = {
        @ApiResponse(responseCode = "200", description = "Resource found",
                    content = @Content(schema = @Schema(implementation = ResourceDTO.class))),
        @ApiResponse(responseCode = "404", description = "Resource not found")
    })
    @GetMapping(value = "/{resourceId}", produces = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<ResourceDTO> getResourceById(
            @Parameter(description = "ID of the resource to retrieve")
            @PathVariable @Min(1) Long resourceId) {
        ResourceDTO resource = resourceService.getResourceById(resourceId);
        return ResponseEntity.ok(resource);
    }

    @Operation(summary = "Download resource file")
    @ApiResponses(value = {
        @ApiResponse(responseCode = "200", description = "File downloaded successfully"),
        @ApiResponse(responseCode = "404", description = "File not found"),
        @ApiResponse(responseCode = "500", description = "Server error during download")
    })
    @GetMapping("/{resourceId}/download")
    public ResponseEntity<?> downloadResource(
            @Parameter(description = "ID of the resource to download")
            @PathVariable @Min(1) Long resourceId) {
        log.info("Initiating download for resource ID: {}", resourceId);
        try {
            Resource resource = resourceService.getResourceEntityById(resourceId);
            String localPath = resource.getLocalPath(); // Assuming you have this field in Resource entity
            
            if (localPath == null || localPath.isEmpty()) {
                // If file is not local, fetch it first
                localPath = resourceFetcherService.downloadResource(resourceId);
            }
            
            File file = new File(localPath);
            if (!file.exists()) {
                return ResponseEntity.notFound().build();
            }

            MediaType mediaType = determineMediaType(file.getName());
            
            return ResponseEntity.ok()
                .contentType(mediaType)
                .header(HttpHeaders.CONTENT_DISPOSITION, 
                       "attachment; filename=\"" + file.getName() + "\"")
                .header(HttpHeaders.CONTENT_LENGTH, String.valueOf(file.length()))
                .body(new FileSystemResource(file));
        } catch (IOException e) {
            log.error("Error downloading resource {}: {}", resourceId, e.getMessage());
            return ResponseEntity.internalServerError()
                .body("Error while processing file download: " + e.getMessage());
        }
    }

    @Operation(summary = "Check if resource exists")
    @Cacheable(value = "resourceExists", key = "#resourceId")
    @GetMapping("/{resourceId}/exists")
    public ResponseEntity<Boolean> checkResourceExists(
            @Parameter(description = "ID of the resource to check")
            @PathVariable @Min(1) Long resourceId) {
        boolean exists = resourceService.resourceExists(resourceId);
        return ResponseEntity.ok(exists);
    }

    @Operation(summary = "Get resource metadata")
    @ApiResponse(responseCode = "200", description = "Successfully retrieved metadata")
    @GetMapping("/{resourceId}/metadata")
    public ResponseEntity<ResourceDTO> getResourceMetadata(
            @Parameter(description = "ID of the resource")
            @PathVariable @Min(1) Long resourceId) {
        ResourceDTO metadata = resourceService.getResourceById(resourceId);
        return ResponseEntity.ok(metadata);
    }

    @Operation(summary = "Upload a new resource")
    @ApiResponses(value = {
        @ApiResponse(responseCode = "201", description = "Resource created successfully"),
        @ApiResponse(responseCode = "400", description = "Invalid input"),
        @ApiResponse(responseCode = "403", description = "Forbidden"),
        @ApiResponse(responseCode = "415", description = "Unsupported file type")
    })
    @PostMapping(value = "/upload", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<ResourceDTO> uploadResource(
            @Parameter(description = "JWT token for authentication", required = true)
            @RequestHeader("Authorization") String token,
            @Parameter(description = "Resource file to upload")
            @RequestParam("file") MultipartFile file,
            @Parameter(description = "Resource category")
            @RequestParam(required = false) String category,
            @Parameter(description = "Resource type")
            @RequestParam(required = false) String resourceType,
            @Parameter(description = "Resource description")
            @RequestParam(required = false) String description) {
        
        if (!utilFunctions.isAdmin(token) && !utilFunctions.isDoctor(token)) {
            return ResponseEntity.status(HttpStatus.FORBIDDEN).build();
        }
        
        log.info("Receiving upload request for file: {}", file.getOriginalFilename());
        
        try {
            ResourceDTO uploadedResource = resourceService.uploadResource(file, category, resourceType, description);
            return ResponseEntity.status(HttpStatus.CREATED).body(uploadedResource);
        } catch (IllegalArgumentException e) {
            log.warn("Invalid upload request: {}", e.getMessage());
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, e.getMessage());
        } catch (IOException e) {
            log.error("File upload failed: {}", e.getMessage());
            throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, "Failed to process file upload");
        }
    }

    /**
     * Determines the MediaType based on file extension.
     * @param fileName the name of the file
     * @return appropriate MediaType for the file
     */
    private MediaType determineMediaType(String fileName) {
        String fileExtension = fileName.substring(fileName.lastIndexOf(".") + 1).toLowerCase();
        switch (fileExtension) {
            case "pdf": return MediaType.APPLICATION_PDF;
            case "jpg":
            case "jpeg": return MediaType.IMAGE_JPEG;
            case "png": return MediaType.IMAGE_PNG;
            case "txt": return MediaType.TEXT_PLAIN;
            default: return MediaType.APPLICATION_OCTET_STREAM;
        }
    }

    @ExceptionHandler(ResourceNotFoundException.class)
    @ResponseStatus(HttpStatus.NOT_FOUND)
    public ResponseEntity<String> handleResourceNotFound(ResourceNotFoundException ex) {
        log.warn("Resource not found: {}", ex.getMessage());
        return ResponseEntity.notFound().build();
    }

    @ExceptionHandler(Exception.class)
    @ResponseStatus(HttpStatus.INTERNAL_SERVER_ERROR)
    public ResponseEntity<String> handleGeneralException(Exception ex) {
        log.error("Unexpected error: ", ex);
        return ResponseEntity.internalServerError()
            .body("An unexpected error occurred");
    }
}
