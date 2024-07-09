package com.thd2020.pasmain.entity;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.persistence.*;
import lombok.Data;

@Entity
@Data
@Schema(description = "科室信息")
public class Department {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Schema(description = "科室ID", example = "1")
    private Long departmentId;

    @Column(nullable = false, length = 100)
    @Schema(description = "科室名称", example = "心内科")
    private String departmentName;

    @ManyToOne
    @JoinColumn(name = "hospital_id", nullable = false)
    @Schema(description = "所属医院")
    private Hospital hospital;

    @Column(length = 20)
    @Schema(description = "科室联系电话", example = "010-12345678")
    private String phone;
}