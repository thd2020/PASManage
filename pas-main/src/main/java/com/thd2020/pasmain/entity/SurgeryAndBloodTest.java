package com.thd2020.pasmain.entity;

import jakarta.persistence.*;
import lombok.Data;

@Entity
@Data
public class SurgeryAndBloodTest {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long recordId;

    @ManyToOne
    @JoinColumn(name = "patient_id", nullable = false)
    private Patient patient;

    private Integer gestationalWeeks;

    @Column(length = 100)
    private String primarySurgeon;

    @Column(length = 100)
    private String assistingSurgeon;

    private Integer preDeliveryBleeding;

    private Integer intraoperativeBleeding;

    private Boolean aorticBalloon;

    private Boolean hysterectomy;

    private Integer redBloodCellsTransfused;

    private Integer hospitalStayDays;

    private Integer plasmaTransfused;

    @Column(precision = 5)
    private Float preoperativeHb;

    @Column(precision = 5)
    private Float preoperativeHct;

    @Column(precision = 5)
    private Float postoperative24hHb;

    @Column(precision = 5)
    private Float postoperative24hHct;

    @Column(length = 100)
    private String postoperativeTransfusionStatus;

    @Lob
    private String notes;
}
