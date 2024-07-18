package com.thd2020.pasmain.entity;

import com.fasterxml.jackson.annotation.JsonIdentityInfo;
import com.fasterxml.jackson.annotation.JsonIdentityReference;
import com.fasterxml.jackson.annotation.ObjectIdGenerators;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.persistence.*;
import lombok.Data;
import org.springframework.core.io.Resource;

import java.time.LocalDateTime;

@Entity
@Data
@Schema(description = "分割/分级结果")
public class PlacentaSegmentationGrading {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Schema(description = "分割/分级结果ID")
    private Long segGradeId;

    @ManyToOne
    @JoinColumn(name = "image_id", nullable = false)
    @JsonIdentityInfo(generator = ObjectIdGenerators.PropertyGenerator.class, property = "imageId")
    @JsonIdentityReference(alwaysAsId = true)
    @Schema(description = "图像ID")
    private Image image;

    @ManyToOne
    @JoinColumn(name = "mask_id", nullable = false)
    @JsonIdentityInfo(generator = ObjectIdGenerators.PropertyGenerator.class, property = "maskId")
    @JsonIdentityReference(alwaysAsId = true)
    @Schema(description = "掩膜ID")
    private Mask mask;

    @Transient
    @Schema(description = "图像文件", example = "1")
    private Resource imageResource;

    @Transient
    @Schema(description = "掩膜文件", example = "1")
    private Resource maskResource;

    @ManyToOne
    @JoinColumn(name = "patient_id", nullable = false)
    @JsonIdentityInfo(generator = ObjectIdGenerators.PropertyGenerator.class, property = "patientId")
    @JsonIdentityReference(alwaysAsId = true)
    @Schema(description = "患者ID")
    private Patient patient;

    @Enumerated(EnumType.STRING)
    @Column
    @Schema(description = "分级结果", example = "normal")
    private Grade grade;

    @Column(precision = 5)
    @Schema(description = "概率", example = "0.95")
    private float probability;

    @Enumerated(EnumType.STRING)
    @Column
    @Schema(description = "总体分级结果", example = "normal")
    private Grade overallGrade;

    @Column(nullable = false)
    @Schema(description = "记录创建时间", example = "2023-07-08T10:15:30")
    private LocalDateTime timestamp;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    @Schema(description = "分割来源", example = "MODEL")
    private SegmentationSource segmentationSource;

    public enum Grade {
        NORMAL, ADHESION, INVASION, PENETRATION
    }

    public enum SegmentationSource {
        MODEL, DOCTOR
    }
}