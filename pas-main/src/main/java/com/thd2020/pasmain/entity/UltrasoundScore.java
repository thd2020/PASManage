package com.thd2020.pasmain.entity;

import jakarta.persistence.*;
import lombok.Data;

import java.time.LocalDateTime;

@Entity
@Data
public class UltrasoundScore {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long scoreId;

    @ManyToOne
    @JoinColumn(name = "patient_id", nullable = false)
    private Patient patient;

    @Column(length = 255)
    private String recordId;

    @Column(nullable = false)
    private LocalDateTime examinationDate;

    private Integer placentalThickness;

    private Integer myometriumInvasion;

    private Integer placentalLacunae;

    private Integer bloodFlowSignals;

    private Integer cervicalShape;

    private Integer suspectedPlacentaLocation;

    private Integer suspectedInvasionRange;

    private Integer placentalPosition;

    private Integer placentalBodyPosition;

    private Integer fetalPosition;

    private Integer totalScore;

    private Integer estimatedBloodLoss;
}