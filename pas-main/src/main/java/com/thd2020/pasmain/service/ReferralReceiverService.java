package com.thd2020.pasmain.service;

import java.util.Optional;

import java.util.List;
import org.springframework.amqp.rabbit.annotation.RabbitListener;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import jakarta.persistence.EntityNotFoundException;

import com.thd2020.pasmain.dto.ReferralBundleDTO;
import com.thd2020.pasmain.entity.Hospital;
import com.thd2020.pasmain.entity.MedicalRecord;
import com.thd2020.pasmain.entity.Patient;
import com.thd2020.pasmain.entity.ReferralRequest;
import com.thd2020.pasmain.entity.SurgeryAndBloodTest;
import com.thd2020.pasmain.entity.UltrasoundScore;
import com.thd2020.pasmain.entity.Patient.ReferralStatus;
import com.thd2020.pasmain.repository.HospitalRepository;
import com.thd2020.pasmain.repository.MedicalRecordRepository;
import com.thd2020.pasmain.repository.PatientRepository;
import com.thd2020.pasmain.repository.ReferralRequestRepository;
import com.thd2020.pasmain.repository.SurgeryAndBloodTestRepository;
import com.thd2020.pasmain.repository.UltrasoundScoreRepository;

@Service
public class ReferralReceiverService {

    @Autowired
    private RabbitTemplate rabbitTemplate;

    @Autowired
    private ReferralRequestRepository referralRequestRepository;

    @Autowired
    private PatientRepository patientRepository;

    @Autowired
    private MedicalRecordRepository medicalRecordRepository;

    @Autowired
    private UltrasoundScoreRepository ultrasoundScoreRepository;

    @Autowired
    private SurgeryAndBloodTestRepository surgeryAndBloodTestRepository;

    @Autowired
    private HospitalRepository hospitalRepository;

    @Autowired
    private RestTemplate restTemplate;

    @Autowired
    private SimpMessagingTemplate messagingTemplate;

    @Value("${app.api-key}")
    private String apiKey;  // Inject API Key from application.properties

    @RabbitListener(queues = "hospital.${server.ip}.queue")
    public void receiveReferralRequest(ReferralRequest referralRequest) {
        // 处理接收到的转诊请求       // Send notification to the front-end via WebSocket
        String notificationMessage = String.format("Referral Request ID %d about patient %s has been received", 
        referralRequest.getRequestId(), 
        referralRequest.getPatient().getName());
        messagingTemplate.convertAndSend("/topic/referralUpdates", notificationMessage);
        referralRequest.setStatus(ReferralRequest.Status.PENDING); // 默认设置为待处理
        referralRequestRepository.save(referralRequest); // 保存到数据库
    }

    public ReferralRequest handleReferralResponse(Long requestId, ReferralRequest.Status status, String reason) {
        ReferralRequest referralRequest = referralRequestRepository.findById(requestId)
                .orElseThrow(() -> new EntityNotFoundException("ReferralRequest not found"));
        referralRequest.setStatus(status);
        referralRequest.setApprovalReason(reason); 
        Patient patient = referralRequest.getPatient();
        if (status == ReferralRequest.Status.APPROVED) {
            patient.setReferralStatus(Patient.ReferralStatus.APPROVED);
        }
        else if (status == ReferralRequest.Status.REJECTED) {
            patient.setReferralStatus(Patient.ReferralStatus.REJECTED);
        }
        // 通过 RabbitMQ 发送通知
        // 查找目标医院的 IP 地址或域名
        String hospitalIp = referralRequest.getFromHospital().getServerIp();
        // 发送请求到目标医院的队列
        // rabbitTemplate.convertAndSend("hospital." + hospitalIp + ".queue", referralRequest);// Send the referral request to the target hospital's REST endpoint
        String destinationUrl = "http://" + hospitalIp + ":8080/api/v1/referrals/response"; // Adjust port as needed
        HttpHeaders headers = new HttpHeaders();
        headers.set("API-Key", apiKey);  // Set the API key in the headers
        headers.setContentType(MediaType.APPLICATION_JSON);
        // Create the request entity with headers and the referral bundle
        HttpEntity<ReferralRequest> requestEntity = new HttpEntity<>(referralRequest, headers);
        // Send the POST request to the other server
        ResponseEntity<String> response = restTemplate.exchange(destinationUrl, HttpMethod.POST, requestEntity, String.class);
        return referralRequestRepository.save(referralRequest);
    }
    
    public Optional<ReferralRequest> getReferralRequestById(Long requestId) {
        return Optional.ofNullable(referralRequestRepository.findById(requestId)
                .orElseThrow(() -> new EntityNotFoundException("Referral request not found with ID: " + requestId)));
    }

    public ReferralRequest receiveReferral(ReferralBundleDTO referralBundle) {
        ReferralRequest referral = referralBundle.getReferral();
        Patient patient = referralBundle.getPatient();
        patient.setFromHospital(referral.getFromHospital());
        patient.setReferralStatus(Patient.ReferralStatus.PENDING);
        List<MedicalRecord> medicalRecords = referralBundle.getMedicalRecords();
        medicalRecords.forEach(medicalRecord -> {
            medicalRecord.setPatient(patient);
        });
        List<UltrasoundScore> ultrasoundScores = referralBundle.getUltrasoundScores();
        ultrasoundScores.forEach(ultrasoundScore -> {
            ultrasoundScore.setPatient(patient);
        });
        List<SurgeryAndBloodTest> surgeryAndBloodTests = referralBundle.getSurgeryAndBloodTests();
        surgeryAndBloodTests.forEach(surgeryAndBloodTest -> {
            surgeryAndBloodTest.setPatient(patient);
        });
        // Set the status to PENDING by default when receiving a new referral request
        referral.setStatus(ReferralRequest.Status.PENDING);
        // Save to the database
        // Check if fromHospital exists in the database
        if (!hospitalRepository.existsById(referral.getFromHospital().getHospitalId())) {
            hospitalRepository.save(referral.getFromHospital());
        }
        patientRepository.save(patient);
        medicalRecordRepository.saveAll(medicalRecords);
        ultrasoundScoreRepository.saveAll(ultrasoundScores);
        surgeryAndBloodTestRepository.saveAll(surgeryAndBloodTests);
        ReferralRequest savedRequest = referralRequestRepository.save(referral);
        // 处理接收到的转诊请求       // Send notification to the front-end via WebSocket
        String notificationMessage = String.format("Referral Request ID %d about patient %s has been received", 
        referral.getRequestId(), 
        referral.getPatient().getName());
        messagingTemplate.convertAndSend("/topic/referralUpdates", notificationMessage);
        return savedRequest;
    }
}

