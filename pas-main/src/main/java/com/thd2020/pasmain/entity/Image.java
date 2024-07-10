package com.thd2020.pasmain.entity;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.persistence.*;
import lombok.Data;
import org.springframework.core.io.Resource;

import java.util.List;

@Entity
@Data
@Schema(description = "影像图像")
public class Image {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Schema(description = "图像ID", example = "1")
    private Long imageId;

    @ManyToOne
    @JoinColumn(name = "record_id", nullable = false)
    @Schema(description = "检测记录ID", example = "record123")
    private ImagingRecord imagingRecord;

    @ManyToOne
    @JoinColumn(name = "patient_id", nullable = false)
    @Schema(description = "患者ID", example = "1")
    private Patient patient;

    @Column(nullable = false, length = 255)
    @Schema(description = "图像文件名", example = "image123.jpg")
    private String imageName;

    @Column(nullable = false, length = 255)
    @Schema(description = "图像文件路径", example = "/path/to/image123.jpg")
    private String imagePath;

    @Transient
    @Schema(description = "图像文件")
    private Resource imageResource;

    @OneToMany(mappedBy = "image", cascade = CascadeType.ALL)
    @Schema(description = "掩膜IDs")
    private List<Mask> masks;
}