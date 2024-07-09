package com.thd2020.pasmain.entity;

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

    @ManyToOne
    @JoinColumn(name = "patient_id", nullable = false)
    @Schema(description = "关联的患者ID", example = "1")
    private Patient patient;

    @Column(length = 255)
    @Schema(description = "检测记录ID", example = "123456789")
    private String recordId;

    @Column(nullable = false)
    @Schema(description = "检查日期", example = "2024-01-01T12:00:00")
    private LocalDateTime examinationDate;

    @Schema(description = "胎盘厚度（小于等于5cm0分，5-8cm1分，8-10cm 2分，大于10cm3分）", example = "2")
    private Integer placentalThickness;

    @Schema(description = "子宫肌层受侵犯程度（胎盘后间隙清晰0分，胎盘后间隙消失，肌层厚度正常1分，肌层厚度变薄未穿透2分，穿透3分）", example = "2")
    private Integer myometriumInvasion;

    @Schema(description = "胎盘内陷窝、沸水征评分（无胎盘陷窝0分；有胎盘陷窝，不论个数，二维图像上无血液流动1分，有静脉样血液流动2分，动脉血液流动即“沸水征”3分）", example = "1")
    private Integer placentalLacunae;

    @Schema(description = "胎盘与子宫肌壁间血流信号评分（血流信号稀疏0分；局部连接成片1分，血流信号与胎盘内陷窝相通2分；血流信号突破子宫浆膜层3分）", example = "3")
    private Integer bloodFlowSignals;

    @Schema(description = "宫颈形态评分（宫颈形态正常，长度大于2.5cm，0分，宫颈整体或局部受侵小于1/3，1分，1/3~2/3：2分，大于2/3：3分）", example = "2")
    private Integer cervicalShape;

    @Schema(description = "可疑胎盘植入位置评分（无：0分；宫底、后壁、侧壁及脐以上前壁段：1分；脐以下前壁：2分；宫颈管内3分）", example = "2")
    private Integer suspectedPlacentaLocation;

    @Schema(description = "可疑植入范围评分（无：0；1个分区:1分；2区：2分，3区以上或位于宫颈管内：3分）", example = "1")
    private Integer suspectedInvasionRange;

    @Schema(description = "胎盘位置评分（中央：1；边缘：2；低置：3）", example = "2")
    private Integer placentalPosition;

    @Schema(description = "胎盘主体位置（前壁：1；后壁：2）", example = "1")
    private Integer placentalBodyPosition;

    @Schema(description = "胎儿位置（头位：1；非头位：2）", example = "1")
    private Integer fetalPosition;

    @Schema(description = "总评分", example = "15")
    private Integer totalScore;

    @Schema(description = "预估出血量（毫升）", example = "500")
    private Integer estimatedBloodLoss;
}