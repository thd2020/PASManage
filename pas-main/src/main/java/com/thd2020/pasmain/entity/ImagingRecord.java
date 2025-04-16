package com.thd2020.pasmain.entity;

import com.fasterxml.jackson.annotation.*;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.persistence.*;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

import org.springframework.core.io.Resource;

import java.time.LocalDateTime;
import java.util.List;

@Entity
@ToString(exclude = {"images"}) // Replace @Data with more specific annotations
@Getter
@Setter
@EqualsAndHashCode(exclude = {"images"})
@Schema(description = "影像检测记录")
public class ImagingRecord {

    @Id
    @Column(length = 255)
    @Schema(description = "检测记录ID", example = "record123")
    private String recordId;

    @ManyToOne
    @JoinColumn(name = "patient_id", nullable = false)
    @JsonIdentityInfo(generator = ObjectIdGenerators.PropertyGenerator.class, property = "patientId")
    @JsonIdentityReference(alwaysAsId = true)
    @Schema(description = "患者ID")
    private Patient patient;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    @Schema(description = "检测类型", example = "CT")
    private TestType testType;

    @Column(nullable = false)
    @Schema(description = "检测日期", example = "2023-07-08T10:15:30")
    private LocalDateTime testDate;

    @Lob
    @Schema(description = "结果描述")
    private String resultDescription;

    @OneToMany(mappedBy = "imagingRecord", cascade = CascadeType.ALL)
    @JsonIdentityInfo(generator = ObjectIdGenerators.PropertyGenerator.class, property = "imageId")
    @JsonIdentityReference(alwaysAsId = true)
    @JsonManagedReference
    @Schema(description = "图像IDs")
    private List<Image> images;

    // @Transient
    // @Schema(description = "图像资源")
    // private List<Resource> imageResources;

    @Column
    @Schema(description = "图像数目", example = "10")
    private int imageCount;

    @Column
    @Schema(description = "图像记录文件路径", example = "/path/to/record123")
    private String path;

    public enum TestType {
        CT, MRI
    }
}