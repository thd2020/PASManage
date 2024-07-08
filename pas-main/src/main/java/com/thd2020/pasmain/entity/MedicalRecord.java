package com.thd2020.pasmain.entity;

import jakarta.persistence.*;
import lombok.Data;

import java.time.LocalDateTime;

@Entity
@Data
public class MedicalRecord {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long recordId;

    @ManyToOne
    @JoinColumn(name = "patient_id", nullable = false)
    private Patient patient;

    @Column(nullable = false)
    private LocalDateTime visitDate;

    @Column(length = 100)
    private String visitType;

    @Column(length = 100)
    private String name;

    private Integer age;

    @Column(precision = 5)
    private Float heightCm;

    @Column(precision = 5)
    private Float preDeliveryWeight;

    @Column(precision = 5)
    private Float afp;

    @Column(precision = 5)
    private Float bHcg;

    @Column(precision = 5)
    private Float ue3;

    @Column(precision = 5)
    private Float inhibinA;

    private Integer gravidity;

    private Integer cSectionCount;

    private Integer currentPregnancyCount;

    private Integer vaginalDeliveryCount;

    private Integer medicalAbortion;

    private Integer surgicalAbortion;

    private Boolean fibroidSurgeryHistory;

    private Boolean uterineSurgeryHistory;

    private Boolean arteryEmbolizationHistory;

    private Boolean pasHistory;

    private Integer assistedReproduction;

    private Boolean hypertension;

    private Boolean diabetes;

    @Enumerated(EnumType.STRING)
    @Column(length = 10)
    private Anemia anemia;

    @Lob
    private String symptoms;

    @Lob
    private String diagnosis;

    @Lob
    private String treatment;

    @ManyToOne
    @JoinColumn(name = "doctor_id")
    private Doctor doctor;

    @Lob
    private String notes;

    public enum Anemia {
        NONE, MILD, MODERATE
    }
}