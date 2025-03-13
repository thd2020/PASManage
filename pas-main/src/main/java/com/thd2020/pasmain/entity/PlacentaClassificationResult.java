package com.thd2020.pasmain.entity;

import jakarta.persistence.*;
import lombok.Data;
import java.time.LocalDateTime;

@Entity
@Data
public class PlacentaClassificationResult {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long classificationId;

    @ManyToOne
    @JoinColumn(name = "image_id", nullable = false)
    private Image image;

    @ManyToOne
    @JoinColumn(name = "patient_id", nullable = false)
    private Patient patient;

    @Column(nullable = true)
    private String classificationPath;

    @Column(nullable = false)
    private LocalDateTime timestamp;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private ClassificationSource classificationSource;

    @Column(nullable = false)
    private double normalProbability;
    
    @Column(nullable = false)
    private double accretaProbability;
    
    @Column(nullable = false)
    private double incretaProbability;
    
    @Column(nullable = false)
    private String predictedType;

    public enum ClassificationSource {
        RESNET, DOCTOR, MLMPAS, MTPAS, VGG16
    }
}
