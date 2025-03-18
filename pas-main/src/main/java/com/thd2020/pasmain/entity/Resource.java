package com.thd2020.pasmain.entity;

import lombok.Data;
import lombok.Getter;
import lombok.Setter;
import jakarta.persistence.*;
import java.time.LocalDateTime;

@Data
@Entity
@Getter
@Setter
@Table(name = "pas_resources")
public class Resource {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String name;
    private String identifier;
    private String sourceUrl;
    private String title;
    private String description;
    private String resourceType; // VIDEO, DOCUMENT, LITERATURE
    private String category; // PREDICTION, PREVENTION, TREATMENT, etc.
    private String fileUrl;
    private String localPath;
    private Long fileSize;
    private String mimeType;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private LocalDateTime timestamp;
}
