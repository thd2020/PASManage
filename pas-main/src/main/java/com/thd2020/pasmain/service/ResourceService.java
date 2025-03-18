package com.thd2020.pasmain.service;

import com.thd2020.pasmain.repository.ResourceRepository;
import com.thd2020.pasmain.dto.ResourceDTO;
import com.thd2020.pasmain.entity.Resource;

import org.springframework.beans.BeanUtils;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.util.StringUtils;
import org.springframework.beans.factory.annotation.Value;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Set;
import java.util.UUID;
import java.util.stream.Collectors;
import java.io.IOException;
import java.time.LocalDateTime;

@Service
public class ResourceService {
    private final ResourceRepository resourceRepository;
    private final ResourceFetcherService resourceFetcherService;

    private static final Set<String> ALLOWED_FILE_TYPES = Set.of(
        "pdf", "doc", "docx", "txt", "jpg", "jpeg", "png"
    );
    
    private static final long MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
    
    @Value("${resource.upload.dir}")
    private String uploadDir;

    public ResourceService(ResourceRepository resourceRepository, 
                         ResourceFetcherService resourceFetcherService) {
        this.resourceRepository = resourceRepository;
        this.resourceFetcherService = resourceFetcherService;
    }

    public List<ResourceDTO> getAllResources() {
        return resourceRepository.findAll().stream()
            .map(this::convertToDTO)
            .collect(Collectors.toList());
    }

    public List<ResourceDTO> getResourcesByCategory(String category) {
        return resourceRepository.findByCategory(category).stream()
            .map(this::convertToDTO)
            .collect(Collectors.toList());
    }

    public List<ResourceDTO> getResourcesByType(String resourceType) {
        return resourceRepository.findByResourceType(resourceType).stream()
            .map(this::convertToDTO)
            .collect(Collectors.toList());
    }

    public void triggerManualFetch() {
        resourceFetcherService.fetchResources();
    }

    public List<ResourceDTO> getLatestResources(int limit) {
        return resourceRepository.findTopByOrderByTimestampDesc(limit)
            .stream()
            .map(this::convertToDTO)
            .collect(Collectors.toList());
    }

    public ResourceDTO uploadResource(MultipartFile file, String category, 
                                   String resourceType, String description) throws IOException {
        validateFile(file);
        
        String fileName = StringUtils.cleanPath(file.getName());
        String fileExtension = getFileExtension(fileName);
        String uniqueFileName = UUID.randomUUID().toString() + "." + fileExtension;
        
        // Create upload directory if it doesn't exist
        Path uploadPath = Paths.get(uploadDir);
        if (!Files.exists(uploadPath)) {
            Files.createDirectories(uploadPath);
        }
        
        // Save file
        Path filePath = uploadPath.resolve(uniqueFileName);
        Files.copy(file.getInputStream(), filePath);
        
        // Create resource entity
        Resource resource = new Resource();
        resource.setName(fileName);
        resource.setLocalPath(filePath.toString());
        resource.setCategory(category != null ? category : "uncategorized");
        resource.setResourceType(determineResourceType(resourceType, fileExtension));
        resource.setDescription(description);
        resource.setFileSize(file.getSize());
        resource.setMimeType(file.getContentType());
        resource.setTimestamp(LocalDateTime.now());
        
        Resource savedResource = resourceRepository.save(resource);
        return convertToDTO(savedResource);
    }

    private void validateFile(MultipartFile file) {
        if (file.isEmpty()) {
            throw new IllegalArgumentException("File cannot be empty");
        }
        
        if (file.getSize() > MAX_FILE_SIZE) {
            throw new IllegalArgumentException("File size exceeds maximum limit");
        }
        
        String fileExtension = getFileExtension(file.getOriginalFilename());
        if (!ALLOWED_FILE_TYPES.contains(fileExtension.toLowerCase())) {
            throw new IllegalArgumentException("File type not allowed");
        }
    }

    private String getFileExtension(String fileName) {
        return fileName.substring(fileName.lastIndexOf(".") + 1).toLowerCase();
    }

    private String determineResourceType(String providedType, String fileExtension) {
        if (providedType != null && !providedType.isEmpty()) {
            return providedType;
        }
        
        return switch (fileExtension.toLowerCase()) {
            case "pdf", "doc", "docx" -> "DOCUMENT";
            case "jpg", "jpeg", "png" -> "IMAGE";
            case "txt" -> "TEXT";
            default -> "OTHER";
        };
    }

    private ResourceDTO convertToDTO(Resource resource) {
        ResourceDTO dto = new ResourceDTO();
        BeanUtils.copyProperties(resource, dto);
        return dto;
    }

    public ResourceDTO getResourceById(Long resourceId) {
        return resourceRepository.findById(resourceId)
            .map(this::convertToDTO)
            .orElse(null);
    }

    public Resource getResourceEntityById(Long resourceId) {
        return resourceRepository.findById(resourceId)
            .orElse(null);
    }

    public boolean resourceExists(Long resourceId) {
        return resourceRepository.existsById(resourceId);
    }
}
