package com.thd2020.pasmain.entity;

import com.fasterxml.jackson.annotation.JsonIdentityInfo;
import com.fasterxml.jackson.annotation.JsonIdentityReference;
import com.fasterxml.jackson.annotation.ObjectIdGenerators;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.persistence.*;
import lombok.Data;

@Entity
@Data
@Schema(description = "医生信息")
public class Doctor {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Schema(description = "医生ID", example = "1")
    private Long doctorId;

    @ManyToOne
    @JoinColumn(name = "user_id", nullable = false)
    @JsonIdentityInfo(generator = ObjectIdGenerators.PropertyGenerator.class, property = "userId")
    @JsonIdentityReference(alwaysAsId = true)
    @Schema(description = "关联的用户")
    private User user;

    @ManyToOne
    @JoinColumn(name = "department_id", nullable = false)
    @JsonIdentityInfo(generator = ObjectIdGenerators.PropertyGenerator.class, property = "departmentId")
    @JsonIdentityReference(alwaysAsId = true)
    @Schema(description = "所属科室")
    private Department department;

    @Column(nullable = false, length = 100)
    @Schema(description = "医生姓名", example = "张三")
    private String name;

    @Column(length = 100)
    @Schema(description = "职称", example = "主任医师")
    private String title;
}