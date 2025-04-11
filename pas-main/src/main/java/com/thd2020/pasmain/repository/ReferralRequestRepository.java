package com.thd2020.pasmain.repository;

import com.thd2020.pasmain.entity.SurgeryAndBloodTest;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import com.thd2020.pasmain.entity.ReferralRequest;
import com.thd2020.pasmain.entity.ReferralRequest.Status;
import com.thd2020.pasmain.entity.Patient;
import com.thd2020.pasmain.entity.Hospital;
import com.thd2020.pasmain.entity.Department;
import com.thd2020.pasmain.entity.Doctor;
import java.util.List;

@Repository
public interface ReferralRequestRepository extends JpaRepository<ReferralRequest, Long> {
    List<ReferralRequest> findByPatient(Patient patient);
    List<ReferralRequest> findByToHospital(Hospital toHospital);
    boolean existsByFromHospital_HospitalIdAndPatient_PatientId(String fromHospitalId, String patientId);
    List<ReferralRequest> findByPatient_PatientId(String patientId);
}

