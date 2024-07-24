package com.thd2020.pasmain.entity;

import com.fasterxml.jackson.annotation.*;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.persistence.*;
import lombok.Data;

import java.time.LocalDate;

@Entity
@Data
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

    @Column
    @Schema(description = "出生日期", example = "1980-01-01")
    private LocalDate birthDate;

    @Column(length = 255)
    @Schema(description = "住址", example = "北京市朝阳区某街道")
    private String address;

    @ManyToOne
    @JoinColumn(name = "doctor_id")
    @JsonBackReference
    @Schema(description = "关联的医生ID")
    private Doctor doctor;

    public enum Gender {
        @Schema(description = "男性")
        MALE,
        @Schema(description = "女性")
        FEMALE,
        @Schema(description = "其他")
        OTHER
    }
}