package com.thd2020.pasmain.entity;

import com.fasterxml.jackson.annotation.JsonBackReference;
import com.fasterxml.jackson.annotation.JsonIdentityInfo;
import com.fasterxml.jackson.annotation.JsonIdentityReference;
import com.fasterxml.jackson.annotation.ObjectIdGenerators;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.persistence.*;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

import org.springframework.core.io.Resource;

import java.util.List;

@Entity
@ToString(exclude = {"imagingRecord", "masks"}) // Replace @Data with more specific annotations
@Getter
@Setter
@EqualsAndHashCode(exclude = {"imagingRecord", "masks"})
@Schema(description = "影像图像")
public class Image {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Schema(description = "图像ID", example = "1")
    private Long imageId;

    @ManyToOne
    @JoinColumn(name = "record_id", nullable = true)
    @JsonIdentityInfo(generator = ObjectIdGenerators.PropertyGenerator.class, property = "recordId")
    @JsonIdentityReference(alwaysAsId = true)
    @JsonBackReference
    @Schema(description = "检测记录ID", example = "record123")
    private ImagingRecord imagingRecord;

    @ManyToOne
    @JoinColumn(name = "patient_id", nullable = false)
    @JsonIdentityInfo(generator = ObjectIdGenerators.PropertyGenerator.class, property = "patientId")
    @JsonIdentityReference(alwaysAsId = true)
    @Schema(description = "患者ID")
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

    @Column
    @Schema(description = "图像文件有效性", example = "EXIST")
    private Availability imageAvail;

    @OneToMany(mappedBy = "image", cascade = CascadeType.ALL)
    @JsonIdentityInfo(generator = ObjectIdGenerators.PropertyGenerator.class, property = "maskId")
    @JsonIdentityReference(alwaysAsId = true)
    @Schema(description = "掩膜IDs")
    private List<Mask> masks;

    public enum Availability {
        EXIST,
        NONEXIST
    }
}