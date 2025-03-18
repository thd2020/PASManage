package com.thd2020.pasmain.service;

import com.thd2020.pasmain.repository.ResourceRepository;
import com.thd2020.pasmain.dto.ResourceDTO;
import com.thd2020.pasmain.entity.Resource;

import org.springframework.beans.BeanUtils;
import org.springframework.stereotype.Service;
import java.util.List;
import java.util.stream.Collectors;

@Service
public class ResourceService {
    private final ResourceRepository resourceRepository;
    private final ResourceFetcherService resourceFetcherService;

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
        return resourceRepository.findTopNByOrderByTimestampDesc(limit)
            .stream()
            .map(this::convertToDTO)
            .collect(Collectors.toList());
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
