package com.thd2020.pasmain.entity;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.persistence.*;
import lombok.Data;

import java.time.LocalDateTime;

@Entity
@Data
@Schema(description = "病历记录")
public class MedicalRecord {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Schema(description = "病历ID", example = "1")
    private Long recordId;

    @ManyToOne
    @JoinColumn(name = "patient_id", nullable = false)
    @Schema(description = "关联的患者ID", example = "1")
    private Patient patient;

    @Column(nullable = false)
    @Schema(description = "就诊日期", example = "2024-01-01T10:00:00")
    private LocalDateTime visitDate;

    @Column(length = 100)
    @Schema(description = "就诊类型", example = "初诊")
    private String visitType;

    @Column(length = 100)
    @Schema(description = "患者姓名", example = "张三")
    private String name;

    @Schema(description = "患者年龄", example = "30")
    private Integer age;

    @Column(precision = 5)
    @Schema(description = "患者身高（cm）", example = "170.5")
    private Float heightCm;

    @Column(precision = 5)
    @Schema(description = "分娩前体重", example = "60.5")
    private Float preDeliveryWeight;

    @Column(precision = 5)
    @Schema(description = "唐筛结果：AFP", example = "2.5")
    private Float afp;

    @Column(precision = 5)
    @Schema(description = "唐筛结果：β-HCG", example = "1.5")
    private Float bHcg;

    @Column(precision = 5)
    @Schema(description = "唐筛结果：uE3", example = "0.5")
    private Float ue3;

    @Column(precision = 5)
    @Schema(description = "唐筛结果：抑制素A", example = "0.7")
    private Float inhibinA;

    @Schema(description = "妊娠次数", example = "2")
    private Integer gravidity;

    @Schema(description = "剖宫次数", example = "1")
    private Integer cSectionCount;

    @Schema(description = "本次妊娠个数", example = "1")
    private Integer currentPregnancyCount;

    @Schema(description = "阴道分娩次数", example = "1")
    private Integer vaginalDeliveryCount;

    @Schema(description = "药物流产次数", example = "0")
    private Integer medicalAbortion;

    @Schema(description = "人工流产次数", example = "0")
    private Integer surgicalAbortion;

    @Schema(description = "是否有子宫肌瘤切除史", example = "false")
    private Boolean fibroidSurgeryHistory;

    @Schema(description = "是否有宫腔手术史", example = "false")
    private Boolean uterineSurgeryHistory;

    @Schema(description = "是否有既往子宫动脉栓塞术史", example = "false")
    private Boolean arteryEmbolizationHistory;

    @Schema(description = "是否有既往PAS病史", example = "false")
    private Boolean pasHistory;

    @Schema(description = "是否有辅助生殖", example = "1")
    private Integer assistedReproduction;

    @Schema(description = "妊娠期高血压（有/无）", example = "false")
    private Boolean hypertension;

    @Schema(description = "妊娠期糖尿病（有/无）", example = "false")
    private Boolean diabetes;

    @Enumerated(EnumType.STRING)
    @Column(length = 10)
    @Schema(description = "贫血情况（无/轻度/中度）", example = "NONE")
    private Anemia anemia;

    @Lob
    @Schema(description = "患者症状描述", example = "无明显症状")
    private String symptoms;

    @Lob
    @Schema(description = "诊断信息", example = "初步诊断为...")
    private String diagnosis;

    @Lob
    @Schema(description = "治疗方案", example = "建议进行...")
    private String treatment;

    @ManyToOne
    @JoinColumn(name = "doctor_id")
    @Schema(description = "关联的医生ID", example = "1")
    private Doctor doctor;

    @Lob
    @Schema(description = "其他备注信息", example = "无特殊备注")
    private String notes;

    public enum Anemia {
        @Schema(description = "无贫血")
        NONE,
        @Schema(description = "轻度贫血")
        MILD,
        @Schema(description = "中度贫血")
        MODERATE
    }
}