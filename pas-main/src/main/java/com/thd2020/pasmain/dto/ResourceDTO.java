package com.thd2020.pasmain.dto;

import lombok.Data;

@Data
public class ResourceDTO {
    private Long id;
    private String title;
    private String description;
    private String resourceType;
    private String category;
    private String fileUrl;
    private String mimeType;
}
