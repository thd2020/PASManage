package com.thd2020.pasmain.entity;

import com.fasterxml.jackson.annotation.JsonIdentityInfo;
import com.fasterxml.jackson.annotation.JsonIdentityReference;
import com.fasterxml.jackson.annotation.ObjectIdGenerators;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.persistence.*;
import lombok.Data;
import org.springframework.core.io.Resource;

@Entity
@Data
@Schema(description = "掩膜信息")
public class Mask {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Schema(description = "掩膜ID", example = "1")
    private Long maskId;

    @ManyToOne
    @JoinColumn(name = "image_id", nullable = false)
    @JsonIdentityInfo(generator = ObjectIdGenerators.PropertyGenerator.class, property = "imageId")
    @JsonIdentityReference(alwaysAsId = true)
    @Schema(description = "图像ID")
    private Image image;

    @Column(nullable = false, length = 255)
    @Schema(description = "分割掩膜路径", example = "/path/to/mask123.png")
    private String segmentationMaskPath;

    @Transient
    @Schema(description = "掩膜文件")
    private Resource maskResource;

    @Column(nullable = true, length = 255)
    @Schema(description = "分割json路径", example = "/path/to/mask123.json")
    private String segmentationJsonPath;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    @Schema(description = "分割来源", example = "MODEL")
    public SegmentationSource segmentationSource;

    public enum SegmentationSource {
        MODEL, DOCTOR
    }
}