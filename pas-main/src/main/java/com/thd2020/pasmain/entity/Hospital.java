package com.thd2020.pasmain.entity;

import com.fasterxml.jackson.annotation.JsonIdentityInfo;
import com.fasterxml.jackson.annotation.JsonIdentityReference;
import com.fasterxml.jackson.annotation.JsonManagedReference;
import com.fasterxml.jackson.annotation.ObjectIdGenerators;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.persistence.*;
import lombok.Data;

import java.util.List;

@Entity
@Data
@Schema(description = "医院信息")
public class Hospital {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Schema(description = "医院ID", example = "1")
    private Long hospitalId;

    @Column(nullable = false, length = 100)
    @Schema(description = "医院名称", example = "北京协和医院")
    private String name;

    @Column(length = 255)
    @Schema(description = "医院地址", example = "北京市东城区东单帅府园1号")
    private String address;

    @Column(precision = 10)
    @Schema(description = "医院经度", example = "116.407526")
    private Double longitude;

    @Column(precision = 10)
    @Schema(description = "医院纬度", example = "39.904030")
    private Double latitude;

    @Column(length = 20)
    @Schema(description = "医院联系电话", example = "010-69156114")
    private String phone;

    @Column(length = 10)
    @Schema(description = "医院邮政编码", example = "100730")
    private String postalCode;

    @Column(length = 50)
    @Schema(description = "医院等级", example = "三级甲等")
    private String grade;

    @Column(length = 50)
    @Schema(description = "医院所在省份", example = "北京市")
    private String province;

    @Column(length = 50)
    @Schema(description = "医院所在城市", example = "北京市")
    private String city;

    @Column(length = 50)
    @Schema(description = "医院所在区/县", example = "东城区")
    private String district;

    @OneToMany(mappedBy = "hospital")
    @JsonIdentityInfo(generator = ObjectIdGenerators.PropertyGenerator.class, property = "departmentId")
    @JsonIdentityReference(alwaysAsId = true)
    @JsonManagedReference
    @Schema(description = "所管科室")
    private List<Department> departments;
}