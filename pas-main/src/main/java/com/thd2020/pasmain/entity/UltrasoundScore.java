package com.thd2020.pasmain.entity;

import com.fasterxml.jackson.annotation.*;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.persistence.*;
import lombok.Data;

import java.time.LocalDateTime;

@Entity
@Data
@Schema(description = "超声评分记录")
public class UltrasoundScore {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Schema(description = "评分ID", example = "1")
    private Long scoreId;

    @ManyToOne(fetch = FetchType.EAGER)
    @JoinColumn(name = "patient_id", nullable = false)
    @JsonIdentityInfo(generator = ObjectIdGenerators.PropertyGenerator.class, property = "patientId")
    @JsonIdentityReference(alwaysAsId = true)
    @JsonBackReference
    @Schema(description = "关联的患者ID")
    @JsonProperty("patient")
    private Patient patient;

    @Column(length = 255)
    @Schema(description = "检测记录ID", example = "123456789")
    private String recordId;

    @Column(nullable = false)
    @Schema(description = "检查日期", example = "2024-01-01T12:00:00")
    private LocalDateTime examinationDate;

    @Schema(description = "胎儿位置（头位：1；非头位：2）", example = "1")
    private Integer fetalPosition;

    @Schema(description = "胎盘位置（中央：1；边缘：2；低置：3）", example = "1")
    private Integer placentalPosition;

    @Schema(description = "胎盘主体位置（前壁：1；后壁：2）", example = "1")
    private Integer placentalBodyPosition;

    @Schema(description = "总评分", example = "15")
    private Integer totalScore;

    @Lob
    @Schema(description = "血流信号评分", example = "局部连接成片")
    private String bloodFlowSignals;

    @Lob
    @Schema(description = "宫颈形态评分", example = "宫颈形态正常")
    private String cervicalShape;

    @Lob
    @Schema(description = "子宫肌层受侵犯程度", example = "胎盘后间隙消失")
    private String myometriumInvasion;

    @Lob
    @Schema(description = "胎盘陷窝/沸水征", example = "静脉样血流动")
    private String placentalLacunae;

    @Lob
    @Schema(description = "胎盘厚度描述", example = "8cm")
    private String placentalThickness;

    @Lob
    @Schema(description = "可疑植入范围", example = "位于宫颈管")
    private String suspectedInvasionRange;

    @Lob
    @Schema(description = "可疑胎盘植入位置", example = "前壁")
    private String suspectedPlacentaLocation;
}
