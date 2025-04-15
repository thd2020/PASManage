package com.thd2020.pasmain.entity;

import com.fasterxml.jackson.annotation.*;
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

    @Schema(description = "分娩孕周", example = "38")
    private String gestationalWeeks;

    @Schema(description = "住院天数", example = "5")
    private Integer hospitalStayDays;

    @Column(length = 100)
    @Schema(description = "主刀医师", example = "张医生")
    private String primarySurgeon;

    @Column(length = 100)
    @Schema(description = "助理医师", example = "李医生")
    private String assistingSurgeon;

    @Column(length = 100)
    @Schema(description = "手术时长", example = "1小时30分钟")
    private String surgeryDuration;

    @Schema(description = "术中出血量（mL）", example = "1200")
    private Integer intraoperativeBleeding;

    @Schema(description = "产前出血量（mL）", example = "400")
    private Integer preDeliveryBleeding;

    @Column(length = 100)
    @Schema(description = "分娩麻醉方式", example = "腰麻")
    private String anesthesiaMethod;

    @Schema(description = "新生儿体重（g）", example = "3200")
    private Float newbornWeight;

    @Schema(description = "新生儿动脉血气PH", example = "7.2")
    private Float arterialPh;

    @Column(length = 100)
    @Schema(description = "Apgar评分", example = "10-10-10")
    private String apgarScore;

    @Schema(description = "术前血红蛋白（g/L）", example = "123.4")
    private Float preoperativeHb;

    @Schema(description = "术前血细胞比容（%）", example = "36.2")
    private Float preoperativeHct;

    @Schema(description = "术后24小时内血红蛋白（g/L）", example = "108.9")
    private Float postoperative24hHb;

    @Schema(description = "术后24小时内血细胞比容（%）", example = "33.7")
    private Float postoperative24hHct;

    @Column(length = 100)
    @Schema(description = "术后输血情况", example = "无")
    private String postoperativeTransfusionStatus;

    @Schema(description = "是否进行宫颈提拉术", example = "true")
    private Boolean cervicalSurgery;

    @Schema(description = "是否双侧子宫动脉结扎", example = "true")
    private Boolean bilateralUterineArteryLigation;

    @Schema(description = "是否双侧卵巢动脉结扎", example = "false")
    private Boolean bilateralOvarianArteryLigation;

    @Column(length = 100)
    @Schema(description = "子宫手术方式", example = "前后壁排式")
    private String uterineSurgeryType;

    @Schema(description = "是否手取胎盘术", example = "true")
    private Boolean placentaRemoval;

    @Schema(description = "是否子宫整形术", example = "true")
    private Boolean uterineReconstruction;

    @Schema(description = "是否输卵管结扎", example = "false")
    private Boolean tubalLigation;

    @Schema(description = "是否使用COOK球囊", example = "true")
    private Boolean cookBalloonSealing;

    @Schema(description = "是否使用腹主动脉球囊", example = "false")
    private Boolean aorticBalloon;

    @Schema(description = "是否进行子宫切除", example = "false")
    private Boolean hysterectomy;

    @Schema(description = "输注红细胞（单位袋）", example = "2")
    private String redBloodCellsTransfused;

    @Schema(description = "输注血浆（单位mL）", example = "500")
    private Integer plasmaTransfused;

    @Schema(description = "剖宫次数", example = "2")
    private Integer cSectionCount;

    @Lob
    @Schema(description = "备注", example = "术后恢复良好")
    private String notes;
}