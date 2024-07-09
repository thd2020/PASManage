package com.thd2020.pasmain.entity;

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

    @Column(nullable = false, length = 20)
    @Schema(description = "身份证号", example = "123456789012345678")
    private String passId;

    @ManyToOne
    @JoinColumn(name = "user_id", nullable = false)
    @Schema(description = "关联的用户ID", example = "2")
    private User user;

    @Column(nullable = false, length = 100)
    @Schema(description = "患者姓名", example = "张三")
    private String name;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    @Schema(description = "性别", example = "MALE")
    private Gender gender;

    @Column(nullable = false)
    @Schema(description = "出生日期", example = "1980-01-01")
    private LocalDate birthDate;

    @Column(nullable = false, length = 255)
    @Schema(description = "住址", example = "北京市朝阳区某街道")
    private String address;

    @ManyToOne
    @JoinColumn(name = "doctor_id")
    @Schema(description = "关联的医生ID", example = "3")
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