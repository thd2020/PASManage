package com.thd2020.pasmain.entity;

import com.fasterxml.jackson.annotation.*;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.persistence.*;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

import java.time.LocalDate;
import java.util.List;

@Entity
@Getter
@Setter
@ToString(exclude = {"user", "doctor", "medicalRecords", "surgeryAndBloodTests", "ultrasoundScores"})
@EqualsAndHashCode(exclude = {"user", "doctor", "medicalRecords", "surgeryAndBloodTests", "ultrasoundScores"})
@Schema(description = "患者信息表")
public class Patient {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Schema(description = "患者ID", example = "1")
    private Long patientId;

    @Column(length = 20)
    @Schema(description = "身份证号", example = "123456789012345678")
    private String passId;

    @OneToOne
    @JoinColumn(name = "user_id", nullable = true)
    @JsonIdentityInfo(generator = ObjectIdGenerators.PropertyGenerator.class, property = "userId")
    @JsonIdentityReference(alwaysAsId = true)
    @Schema(description = "关联的用户ID", example = "2")
    private User user;

    @Column(nullable = false, length = 100)
    @Schema(description = "患者姓名", example = "张三")
    private String name;

    @Enumerated(EnumType.STRING)
    @Column
    @Schema(description = "性别", example = "MALE")
    private Gender gender;

    @OneToOne
    @JoinColumn(name = "from_hospital", nullable = true)
    @JsonIdentityInfo(generator = ObjectIdGenerators.PropertyGenerator.class, property = "hospitalId")
    @JsonIdentityReference(alwaysAsId = true)
    @Schema(description = "来自医院", example = "1980-01-01")
    private Hospital fromHospital;

    @Enumerated(EnumType.STRING)
    @Column
    @Schema(description = "转院状态", example = "PENDING")
    private ReferralStatus referralStatus;

    @Column
    @Schema(description = "出生日期")
    private LocalDate birthDate;

    @Column(length = 255)
    @Schema(description = "住址", example = "北京市朝阳区某街道")
    private String address;

    @ManyToOne
    @JoinColumn(name = "doctor_id")
    @JsonBackReference
    @Schema(description = "关联的医生ID")
    private Doctor doctor;

    @OneToMany(mappedBy = "patient", cascade = CascadeType.ALL)
    @JsonManagedReference
    @Schema(description = "病历记录ID列表")
    private List<MedicalRecord> medicalRecords;

    @OneToMany(mappedBy = "patient", cascade = CascadeType.ALL)
    @JsonManagedReference
    @Schema(description = "手术和血液检查记录ID列表")
    private List<SurgeryAndBloodTest> surgeryAndBloodTests;

    @OneToMany(mappedBy = "patient", cascade = CascadeType.ALL)
    @JsonManagedReference
    @Schema(description = "超声评分记录ID列表")
    private List<UltrasoundScore> ultrasoundScores;

    public enum Gender {
        @Schema(description = "男性")
        MALE,
        @Schema(description = "女性")
        FEMALE,
        @Schema(description = "其他")
        OTHER
    }

    public enum ReferralStatus {
        PENDING,
        REJECTED,
        APPROVED
    }
}