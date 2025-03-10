package com.thd2020.pasmain.entity;

import com.fasterxml.jackson.annotation.JsonIdentityInfo;
import com.fasterxml.jackson.annotation.JsonIdentityReference;
import com.fasterxml.jackson.annotation.ObjectIdGenerators;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.persistence.*;
import lombok.Data;

import java.time.LocalDateTime;

@Entity
@Data
@JsonIdentityInfo(generator = ObjectIdGenerators.PropertyGenerator.class, property = "requestId")
@Schema(description = "Represents a referral request made between hospitals")
public class ReferralRequest {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Schema(description = "Unique identifier for the referral request", example = "1")
    private Long requestId;

    @ManyToOne(optional = false)
    @JoinColumn(name = "patient_id", nullable = false)
    @JsonIdentityReference(alwaysAsId = true)  // This avoids full serialization of Patient object
    @Schema(description = "The patient being referred", example = "101")
    private Patient patient;

    @ManyToOne(optional = true)
    @JoinColumn(name = "from_user_id", nullable = true)
    @JsonIdentityInfo(generator = ObjectIdGenerators.PropertyGenerator.class, property = "userId")
    @JsonIdentityReference(alwaysAsId = true)
    @Schema(description = "The user initiating the referral", example = "15")
    private User fromUser;

    @ManyToOne(optional = false)
    @JoinColumn(name = "to_hospital_id", nullable = false)
    @JsonIdentityReference(alwaysAsId = true)
    @Schema(description = "The hospital to which the patient is referred", example = "7")
    private Hospital toHospital;

    @ManyToOne(optional = false)
    @JoinColumn(name = "from_hospital_id", nullable = false)
    @JsonIdentityReference(alwaysAsId = true)
    @Schema(description = "The hospital from which the referral is initiated", example = "3")
    private Hospital fromHospital;

    @Column(nullable = false)
    @Schema(description = "Date and time the referral was created", example = "2024-09-09T10:00:00")
    private LocalDateTime requestDate;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    @Schema(description = "The status of the referral", example = "PENDING")
    private Status status;

    @Column(columnDefinition = "TEXT")
    @Schema(description = "Reason for the referral", example = "Patient requires specialized care")
    private String referralReason;

    @Column(columnDefinition = "TEXT")
    @Schema(description = "Reason for approval or rejection", example = "Specialist available")
    private String approvalReason;

    @Column(length = 100)
    @Schema(description = "Name of the doctor who initiated the referral", example = "Dr. Smith")
    private String doctorName;

    @Column(length = 100)
    @Schema(description = "Title of the doctor who initiated the referral", example = "Chief Physician")
    private String doctorTitle;

    @Column(length = 100)
    @Schema(description = "Department of the doctor who initiated the referral", example = "Obstetrics")
    private String departmentName;

    // Enum representing the status of the referral
    public enum Status {
        PENDING,
        APPROVED,
        REJECTED
    }
}
