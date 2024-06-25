package com.thd2020.pasmain.entity;

import jakarta.persistence.*;
import lombok.Data;

@Entity
@Data
public class Hospital {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long hospitalId;

    @Column(nullable = false, length = 100)
    private String name;

    @Column(length = 255)
    private String address;

    @Column(precision = 10)
    private Double longitude;

    @Column(precision = 10)
    private Double latitude;

    @Column(length = 20)
    private String phone;

    @Column(length = 10)
    private String postalCode;

    @Column(length = 50)
    private String grade;

    @Column(length = 50)
    private String province;

    @Column(length = 50)
    private String city;

    @Column(length = 50)
    private String district;
}
