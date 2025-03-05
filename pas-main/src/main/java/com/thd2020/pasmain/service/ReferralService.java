package com.thd2020.pasmain.service;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

import org.springframework.amqp.rabbit.annotation.RabbitListener;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClientException;
import org.springframework.web.client.RestTemplate;
import org.springframework.messaging.simp.SimpMessagingTemplate;

import com.thd2020.pasmain.repository.DoctorRepository;
import com.thd2020.pasmain.repository.HospitalRepository;
import com.thd2020.pasmain.repository.ReferralRequestRepository;
import com.thd2020.pasmain.util.JwtUtil;
import com.thd2020.pasmain.util.UtilFunctions;

import jakarta.persistence.EntityNotFoundException;

import com.thd2020.pasmain.dto.ReferralBundleDTO;
import com.thd2020.pasmain.entity.Doctor;
import com.thd2020.pasmain.entity.Hospital;
import com.thd2020.pasmain.entity.Image;
import com.thd2020.pasmain.entity.Mask;
import com.thd2020.pasmain.entity.MedicalRecord;
import com.thd2020.pasmain.entity.Patient;
import com.thd2020.pasmain.entity.PlacentaSegmentationGrading;
import com.thd2020.pasmain.entity.ReferralRequest;
import com.thd2020.pasmain.entity.SurgeryAndBloodTest;
import com.thd2020.pasmain.entity.UltrasoundScore;
import com.thd2020.pasmain.entity.User;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;

@Service
public class ReferralService {

    @Autowired
    private ReferralRequestRepository referralRequestRepository;

    @Autowired
    private RabbitTemplate rabbitTemplate;

    @Autowired
    private HospitalRepository hospitalRepository;

    @Autowired
    private SimpMessagingTemplate messagingTemplate;

    @Autowired
    private UserService userService;

    @Autowired
    private PatientService patientService;

    @Autowired
    private PRInfoService prInfoService;

    @Autowired
    private ImagingService imagingService;

    @Autowired
    private RestTemplate restTemplate;

    @Autowired
    private UtilFunctions utilFunctions;

    @Autowired
    private JwtUtil jwtUtil;

    @Autowired
    private DocInfoService docInfoService;

    @Autowired
    private DoctorRepository doctorRepository;

    @Autowired
    private ObjectMapper objectMapper;

    @Value("${app.api-key}")
    private String apiKey;  // Inject API Key from application.properties

    public ReferralRequest sendReferralRequest(ReferralRequest request, String token) throws JsonProcessingException {
        // 保存转诊请求到数据库
        Long userId = jwtUtil.extractUserId(token.substring(7));
        User user =  userService.getUserById(userId).get();
        request.setRequestDate(LocalDateTime.now());
        request.setStatus(ReferralRequest.Status.PENDING);
        Hospital hospital = new Hospital();
        if (user.getRole() == User.Role.T_DOCTOR || user.getRole() == User.Role.B_DOCTOR) {
            Doctor doctor = doctorRepository.findByUser_UserId(userId).get();
            request.setFromUser(user);
            hospital = doctor.getDepartment().getHospital();
            request.setFromHospital(hospital);
        }
        else if (user.getRole() == User.Role.ADMIN) {
            if (user.getHospital() != null){
                request.setFromUser(user);
                hospital = user.getHospital();
                request.setFromHospital(hospital);
            }
        }
        Long hospitalId = hospital.getHospitalId();
        Long patientId = request.getPatient().getPatientId();
        if (referralRequestRepository.existsByFromHospital_HospitalIdAndPatient_PatientId(hospitalId, patientId)) {
            throw new IllegalArgumentException("Referral request already exists for this patient and hospital.");
        }
        referralRequestRepository.save(request);

        if (referralRequestRepository.existsByFromHospital_HospitalIdAndPatient_PatientId(hospitalId, patientId)) {
            return null;
        }

        ReferralBundleDTO referralBundleDTO = new ReferralBundleDTO();

        // Fetching the necessary entities
        Patient patient = patientService.getPatient(patientId);
        List<MedicalRecord> MRs = prInfoService.findMedicalRecordIdsByPatientId(patientId);
        List<SurgeryAndBloodTest> sbTests = prInfoService.findSBRecordIdsByPatientId(patientId);
        List<UltrasoundScore> ultrasoundScores = prInfoService.findUltrasoundScoreIdsByPatientId(patientId);
        List<PlacentaSegmentationGrading> gradingResults = imagingService.findGradingByPatientId(patientId);

        // Populate the DTO with the fetched entities
        referralBundleDTO.setReferral(request);
        referralBundleDTO.setPatient(patient);
        referralBundleDTO.setMedicalRecords(MRs);
        referralBundleDTO.setUltrasoundScores(ultrasoundScores);
        referralBundleDTO.setSurgeryAndBloodTests(sbTests);
        referralBundleDTO.setGradings(gradingResults);


        // 查找目标医院的 IP 地址或域名
        Hospital toHospital = hospitalRepository.getReferenceById(request.getToHospital().getHospitalId());
        String hospitalIp = toHospital.getServerIp();

        // 发送请求到目标医院的队列
        // rabbitTemplate.convertAndSend("referral.exchange", "hospital." + hospitalIp + ".queue", referralBundleDTO);
        String destinationUrl = "http://" + hospitalIp + ":8080/api/v1/referrals/receive"; // Adjust port as needed
        HttpHeaders headers = new HttpHeaders();
        headers.set("API-Key", apiKey);  // Set the API key in the headers
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<ReferralBundleDTO> requestEntity = new HttpEntity<ReferralBundleDTO>(referralBundleDTO, headers);
        ResponseEntity<String> response = restTemplate.exchange(destinationUrl, HttpMethod.POST, requestEntity, String.class);
        return request;
    }

    @RabbitListener(queues = "hospital.${server.ip}.queue")  // Listen for responses from the receiving hospital
    public ReferralRequest receiveReferralResponse(ReferralRequest referralRequest) {
        // Fetch the original referral request from the database
        ReferralRequest existingRequest = referralRequestRepository.findById(referralRequest.getRequestId())
                .orElseThrow(() -> new EntityNotFoundException("ReferralRequest not found"));

        existingRequest.setStatus(referralRequest.getStatus());
        existingRequest.setApprovalReason(referralRequest.getApprovalReason());

        ReferralRequest updatedRequest = referralRequestRepository.save(existingRequest);

        // Send notification to the front-end via WebSocket
        String notificationMessage = String.format("Referral Request ID %d has been %s with reason: %s", 
                existingRequest.getRequestId(), 
                existingRequest.getStatus(), 
                existingRequest.getApprovalReason());
        
        messagingTemplate.convertAndSend("/topic/referralUpdates", notificationMessage);
        return updatedRequest;
    }

    public Optional<ReferralRequest> getReferralRequestById(Long requestId) {
        return referralRequestRepository.findById(requestId);
    }

    public List<ReferralRequest> getAllReferralRequest() {
        return referralRequestRepository.findAll();
    }
}

