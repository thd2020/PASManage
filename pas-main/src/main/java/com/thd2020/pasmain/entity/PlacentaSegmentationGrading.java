package com.thd2020.pasmain.entity;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.persistence.*;
import lombok.Data;

import java.time.LocalDateTime;

@Entity
@Data
@Schema(description = "分割/分级结果")
public class PlacentaSegmentationGrading {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Schema(description = "分割/分级结果ID", example = "1")
    private Long segGradeId;

    @ManyToOne
    @JoinColumn(name = "image_id", nullable = false)
    @Schema(description = "图像ID", example = "1")
    private Image image;

    @ManyToOne
    @JoinColumn(name = "mask_id", nullable = false)
    @Schema(description = "掩膜ID", example = "1")
    private Mask mask;

    @ManyToOne
    @JoinColumn(name = "patient_id", nullable = false)
    @Schema(description = "患者ID", example = "1")
    private Patient patient;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    @Schema(description = "分级结果", example = "normal")
    private Grade grade;

    @Column(nullable = false, precision = 5, scale = 4)
    @Schema(description = "概率", example = "0.95")
    private float probability;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
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