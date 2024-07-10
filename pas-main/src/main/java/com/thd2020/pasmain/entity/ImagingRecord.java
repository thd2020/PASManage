package com.thd2020.pasmain.entity;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.persistence.*;
import lombok.Data;
import org.springframework.core.io.Resource;

import java.time.LocalDateTime;
import java.util.List;

@Entity
@Data
@Schema(description = "影像检测记录")
public class ImagingRecord {

    @Id
    @Column(length = 255)
    @Schema(description = "检测记录ID", example = "record123")
    private String recordId;

    @ManyToOne
    @JoinColumn(name = "patient_id", nullable = false)
    @Schema(description = "患者ID", example = "1")
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
    @Schema(description = "图像IDs")
    private List<Image> images;

    @Transient
    @Schema(description = "图像资源")
    private List<Resource> imageResources;

    @Column
    @Schema(description = "图像数目", example = "10")
    private int imageCount;

    @Column
    @Schema(description = "标注图像数目", example = "5")
    private int labelCount;

    public enum TestType {
        CT, MRI
    }
}