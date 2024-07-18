package com.thd2020.pasmain.entity;

import com.fasterxml.jackson.annotation.JsonIdentityInfo;
import com.fasterxml.jackson.annotation.JsonIdentityReference;
import com.fasterxml.jackson.annotation.ObjectIdGenerators;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.persistence.*;
import lombok.Data;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.userdetails.UserDetails;

import java.time.LocalDateTime;
import java.util.Collection;

@Entity
@Data
@Schema(description = "用户注册信息")
public class User implements UserDetails {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Schema(description = "用户ID", example = "1")
    private Long userId;

    @OneToOne
    @JoinColumn(name = "patient_id")
    @JsonIdentityInfo(generator = ObjectIdGenerators.PropertyGenerator.class, property = "doctorId")
    @JsonIdentityReference(alwaysAsId = true)
    @Schema(description = "关联病人id", example = "1")
    private Patient patient;

    @OneToOne
    @JoinColumn(name = "doctor_id")
    @JsonIdentityInfo(generator = ObjectIdGenerators.PropertyGenerator.class, property = "doctorId")
    @JsonIdentityReference(alwaysAsId = true)
    @Schema(description = "关联医生id", example = "1")
    private Doctor doctor;

    @Column(nullable = false, length = 50)
    @Schema(description = "用户名", example = "john_doe")
    private String username;

    @Column(nullable = false, length = 255)
    @Schema(description = "用户密码", example = "password123")
    private String password;

    @Column(length = 100)
    @Schema(description = "电子邮件地址", example = "john.doe@example.com")
    private String email;

    @Column(length = 20)
    @Schema(description = "联系电话", example = "1234567890")
    private String phone;

    @ManyToOne
    @JoinColumn(name = "hospital_id")
    @JsonIdentityInfo(generator = ObjectIdGenerators.PropertyGenerator.class, property = "doctorId")
    @JsonIdentityReference(alwaysAsId = true)
    @Schema(description = "隶属医院id", example = "1")
    private Hospital hospital;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    @Schema(description = "用户角色", example = "PATIENT")
    private Role role;

    @Column(nullable = false)
    @Schema(description = "创建时间", example = "2024-06-25T12:34:56")
    private LocalDateTime createdAt;

    @Schema(description = "最后登录时间", example = "2024-07-01T15:30:00")
    private LocalDateTime lastLogin;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    @Schema(description = "账户状态", example = "ACTIVE")
    private Status status;

    @Override
    public Collection<? extends GrantedAuthority> getAuthorities() {
        return new java.util.ArrayList<>();
    }

    @Enumerated(EnumType.STRING)
    @Schema(description = "认证提供者", example = "GOOGLE")
    private Provider provider;

    public enum Role {
        T_DOCTOR,
        B_DOCTOR,
        PATIENT,
        ADMIN
    }

    public enum Status {
        ACTIVE,
        INACTIVE,
        BANNED
    }

    public enum Provider {
        LOCAL, GOOGLE
    }
}