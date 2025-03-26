package com.thd2020.pasmain.entity;

import com.fasterxml.jackson.annotation.JsonBackReference;
import com.fasterxml.jackson.annotation.JsonIdentityInfo;
import com.fasterxml.jackson.annotation.JsonIdentityReference;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.ObjectIdGenerators;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.persistence.*;
import lombok.Data;

@Entity
@Data
@Schema(description = "手术和血液检查记录")
    public class SurgeryAndBloodTest {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Schema(description = "记录ID", example = "1")
    private Long recordId;

    @ManyToOne(fetch = FetchType.EAGER)
    @JoinColumn(name = "patient_id", nullable = false)
    @JsonIdentityInfo(generator = ObjectIdGenerators.PropertyGenerator.class, property = "patientId", scope = Patient.class)
    @JsonIdentityReference(alwaysAsId = true)
    @JsonBackReference
    @Schema(description = "关联的患者ID", ref = "Patient")
    @JsonProperty("patient")
    private Patient patient;

    @Schema(description = "分娩孕周", example = "39")
    private Integer gestationalWeeks;

    @Column(length = 100)
    @Schema(description = "主刀医师", example = "张医生")
    private String primarySurgeon;

    @Column(length = 100)
    @Schema(description = "助理医师", example = "李医生")
    private String assistingSurgeon;

    @Schema(description = "产前出血量（毫升）", example = "500")
    private Integer preDeliveryBleeding;

    @Schema(description = "术中出血量（毫升）", example = "1000")
    private Integer intraoperativeBleeding;

    @Schema(description = "是否使用腹主动脉球囊", example = "true")
    private Boolean aorticBalloon;

    @Schema(description = "是否进行子宫切除", example = "false")
    private Boolean hysterectomy;

    @Schema(description = "输注红细胞量（单位：袋）", example = "2")
    private Integer redBloodCellsTransfused;

    @Schema(description = "住院天数", example = "5")
    private Integer hospitalStayDays;

    @Schema(description = "输注血浆量（单位：袋）", example = "3")
    private Integer plasmaTransfused;

    @Column(precision = 5)
    @Schema(description = "术前血红蛋白（g/L）", example = "120.5")
    private Float preoperativeHb;

    @Column(precision = 5)
    @Schema(description = "术前血细胞比容（%）", example = "35.0")
    private Float preoperativeHct;

    @Column(precision = 5)
    @Schema(description = "术后24小时内血红蛋白（g/L）", example = "110.5")
    private Float postoperative24hHb;

    @Column(precision = 5)
    @Schema(description = "术后24小时内血细胞比容（%）", example = "32.5")
    private Float postoperative24hHct;

    @Column(length = 100)
    @Schema(description = "术后输血情况", example = "无")
    private String postoperativeTransfusionStatus;

    @Lob
    @Schema(description = "备注信息", example = "手术顺利，患者恢复良好")
    private String notes;
}