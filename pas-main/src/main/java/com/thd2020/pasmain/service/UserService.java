package com.thd2020.pasmain.service;

import com.thd2020.pasmain.dto.ApiResponse;
import com.thd2020.pasmain.dto.OAuthRequest;
import com.thd2020.pasmain.dto.UserRegistrationRequest;
import com.thd2020.pasmain.entity.Doctor;
import com.thd2020.pasmain.entity.Hospital;
import com.thd2020.pasmain.entity.Patient;
import com.thd2020.pasmain.repository.DoctorRepository;
import com.thd2020.pasmain.repository.HospitalRepository;
import com.thd2020.pasmain.repository.PatientRepository;
import io.swagger.v3.oas.annotations.Parameter;
import jakarta.persistence.EntityNotFoundException;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import com.thd2020.pasmain.entity.User;
import com.thd2020.pasmain.repository.UserRepository;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestHeader;

import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.stream.Collectors;

@Service
public class UserService {
    @Autowired
    private UserRepository userRepository;

    @Autowired
    private PatientRepository patientRepository;

    @Autowired
    private DoctorRepository doctorRepository;

    @Autowired
    private PasswordEncoder passwordEncoder;

    @Autowired
    private TokenBlacklistService tokenBlacklistService;

    @Autowired
    private SmsService smsService;

    @Autowired
    private HospitalRepository hospitalRepository;

    private final Map<String, String> verificationCodes = new HashMap<>();

    public void validateRegistration(UserRegistrationRequest request) {
        if (userRepository.findByUsername(request.getUsername()).isPresent()) {
            throw new IllegalArgumentException("Username already exists");
        }
        if (request.getEmail() != null && !request.getEmail().isEmpty() 
            && userRepository.findByEmail(request.getEmail()).isPresent()) {
            throw new IllegalArgumentException("Email already exists");
        }
        if (request.getPhone() != null && !request.getPhone().isEmpty() 
            && userRepository.findByPhone(request.getPhone()).isPresent()) {
            throw new IllegalArgumentException("Phone number already exists");
        }
        if (request.getRole() == User.Role.PATIENT) {
            if (patientRepository.findById(request.getPassId()) != null) {
                throw new IllegalArgumentException("PassId already exists");
            }
        } else if (request.getRole() == User.Role.T_DOCTOR || request.getRole() == User.Role.B_DOCTOR) {
            if (doctorRepository.findByPassId(request.getPassId()) != null) {
                throw new IllegalArgumentException("PassId already exists");
            }
        }
    }

    // 用户注册
    public User registerUser(UserRegistrationRequest request) {
        // Add validation before registration
        validateRegistration(request);
        
        // First, save the user
        User user = new User();
        user.setUsername(request.getUsername());
        user.setPassword(passwordEncoder.encode(request.getPassword()));
        user.setEmail(request.getEmail());
        user.setPhone(request.getPhone());
        user.setRole(request.getRole());
        user.setCreatedAt(LocalDateTime.now());
        user.setStatus(User.Status.ACTIVE);
        user.setProvider(User.Provider.LOCAL);
        
        // Save the user first to get the ID
        user = userRepository.save(user);
    
        // Then handle patient registration
        if (request.getRole() == User.Role.PATIENT) {
            Patient existingPatient = patientRepository.findById(request.getPassId()).get();
            if (existingPatient != null) {
                // Link existing patient to saved user
                existingPatient.setUser(user);
                patientRepository.save(existingPatient);
            } else {
                // Create new patient with saved user
                Patient patient = new Patient();
                patient.setUser(user);
                patient.setName(request.getName());
                patient.setPatientId(request.getPassId());
                patientRepository.save(patient);
            }
        }
        // Similar pattern for doctor registration
        else if (request.getRole() == User.Role.T_DOCTOR || request.getRole() == User.Role.B_DOCTOR) {
            Doctor existingDoctor = doctorRepository.findByPassId(request.getPassId());
            if (existingDoctor != null) {
                existingDoctor.setUser(user);
                doctorRepository.save(existingDoctor);
            } else {
                Doctor doctor = new Doctor();
                doctor.setUser(user);
                doctor.setName(request.getName());
                doctor.setPassId(request.getPassId());
                doctorRepository.save(doctor);
            }
        }
    
        return user;
    }

    public User processOAuthPostLogin(String email, String userName) {
        Optional<User> existUser = userRepository.findByEmail(email);
        if (existUser.isEmpty()) {
            User user = new User();
            user.setUsername(userName);
            user.setEmail(email);
            user.setProvider(User.Provider.GOOGLE);
            user.setCreatedAt(LocalDateTime.now());
            user.setStatus(User.Status.ACTIVE); // 默认设置为ACTIVE
            String rawPassword = String.format("%s:%s:%s", userName, email, LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMddHHmmss")));
            user.setPassword(passwordEncoder.encode(rawPassword));
            user.setRole(User.Role.PATIENT);
            return userRepository.save(user);
        }
        else{
            return existUser.get();
        }
    }


    // 用户登录
    public Optional<User> loginUser(Long userId) {
        Optional<User> userOptional = userRepository.findById(userId);
        if (userOptional.isPresent()) {
            User user = userOptional.get();
            user.setLastLogin(LocalDateTime.now());
            user = userRepository.save(user);
            return Optional.of(user);
        } else {
            return Optional.empty();
        }
    }

    public Optional<User> loginWithPhone(String phone, String password) {
        Optional<User> userOpt = userRepository.findByPhone(phone);
        if (userOpt.isPresent() && passwordEncoder.matches(password, userOpt.get().getPassword())) {
            User user = userOpt.get();
            user.setLastLogin(LocalDateTime.now());
            return Optional.of(userRepository.save(user));
        }
        return Optional.empty();
    }

    public Optional<User> loginWithEmail(String email, String password) {
        Optional<User> userOpt = userRepository.findByEmail(email);
        if (userOpt.isPresent() && passwordEncoder.matches(password, userOpt.get().getPassword())) {
            User user = userOpt.get();
            user.setLastLogin(LocalDateTime.now());
            return Optional.of(userRepository.save(user));
        }
        return Optional.empty();
    }

    public Optional<User> loginWithPassId(String passId, String password) {
        Optional<Patient> patientOpt = patientRepository.findById(passId);
        Optional<Doctor> doctorOpt = Optional.ofNullable(doctorRepository.findByPassId(passId));
        
        if (patientOpt.isPresent()) {
            User user = patientOpt.get().getUser();
            if (passwordEncoder.matches(password, user.getPassword())) {
                user.setLastLogin(LocalDateTime.now());
                return Optional.of(userRepository.save(user));
            }
        } else if (doctorOpt.isPresent()) {
            User user = doctorOpt.get().getUser();
            if (passwordEncoder.matches(password, user.getPassword())) {
                user.setLastLogin(LocalDateTime.now());
                return Optional.of(userRepository.save(user));
            }
        }
        return Optional.empty();
    }

    public void generateAndSendCode(String phone) throws IOException {
        smsService.sendSms(phone);
    }

    public boolean verifyCode(String phone, String code) throws IOException {
        return smsService.verifySms(phone, code);
    }

    public User findOrCreateUserByPhone(String phone) {
        User user = userRepository.findByPhone(phone).isPresent()?userRepository.findByPhone(phone).get():null;
        if (user == null) {
            user = new User();
            user.setCreatedAt(LocalDateTime.now());
            user.setStatus(User.Status.ACTIVE); // 默认设置为ACTIVE
            user.setRole(User.Role.PATIENT);
            user.setPassword(passwordEncoder.encode(phone + LocalDateTime.now())); // 使用phone和当前时间生成密码
            userRepository.save(user);
        }
        return user;
    }

    // 获取用户信息
    public Optional<User> getUserById(Long userId) {
        return userRepository.findById(userId);
    }
    public Optional<User> getUserByUsername(String username) { return userRepository.findByUsername(username); }


    // 更新用户信息
    public Optional<User> updateUser(Long userId, User updatedUser) {
        return userRepository.findById(userId).map(user -> {
            if (updatedUser.getUsername() != null && !updatedUser.getUsername().isEmpty()) {
                user.setUsername(updatedUser.getUsername());
            }
            if (updatedUser.getEmail() != null && !updatedUser.getEmail().isEmpty()) {
                user.setEmail(updatedUser.getEmail());
            }
            if (updatedUser.getPhone() != null && !updatedUser.getPhone().isEmpty()) {
                user.setPhone(updatedUser.getPhone());
            }
            if (updatedUser.getPassword() != null && !updatedUser.getPassword().isEmpty()) {
                user.setPassword(passwordEncoder.encode(updatedUser.getPassword()));
            }
            if (updatedUser.getRole() != null) {
                user.setRole(updatedUser.getRole());
            }
            if (updatedUser.getStatus() != null) {
                user.setStatus(updatedUser.getStatus());
            }
            Hospital hospital = hospitalRepository.findById(updatedUser.getHospital().getHospitalId())
                    .orElseThrow(() -> new EntityNotFoundException("Hospital not found"));
            user.setHospital(hospital);
            return userRepository.save(user);
        });
    }

    // 更改密码
    public boolean changePassword(Long userId, String oldPassword, String newPassword) {
        Optional<User> userOptional = userRepository.findById(userId);
        if (userOptional.isPresent()) {
            User user = userOptional.get();
            if (passwordEncoder.matches(oldPassword, user.getPassword())) {
                user.setPassword(passwordEncoder.encode(newPassword));
                userRepository.save(user);
                return true;
            }
        }
        return false;
    }

    // 重置密码
    public boolean resetPassword(String email, String newPassword) {
        Optional<User> userOptional = userRepository.findByEmail(email);
        if (userOptional.isPresent()) {
            User user = userOptional.get();
            user.setPassword(passwordEncoder.encode(newPassword));
            userRepository.save(user);
            return true;
        }
        return false;
    }

    // 删除用户
    public boolean deleteUser(Long userId) {
        if (userRepository.existsById(userId)) {
            userRepository.deleteById(userId);
            return true;
        }
        return false;
    }

    // 用户登出
    public void logoutUser(String token) {
        // 将当前用户的token添加到黑名单
        tokenBlacklistService.blacklistToken(token);
    }

    // 获取所有用户列表（管理员权限）
    public List<User> getAllUsers() {
        return userRepository.findAll();
    }

    // 分配用户角色（管理员权限）
    public Optional<User> assignRole(Long userId, User.Role role) {
        return userRepository.findById(userId).map(user -> {
            user.setRole(role);
            return userRepository.save(user);
        });
    }

    // 通过姓名找所有匹配的id
    public Long findUserIdsByUsername(String username) {
        Optional<User> user = userRepository.findByUsername(username);
        return user.map(User::getUserId).orElse(null);
    }

    public ApiResponse<?> getUserDetails(Long userId) {
        Optional<User> userOpt = userRepository.findById(userId);
        if (userOpt.isPresent()) {
            User user = userOpt.get();
            if (user.getRole() == User.Role.PATIENT) {
                Optional<Patient> patientOpt = patientRepository.findByUser_UserId(userId);
                if (patientOpt.isPresent()) {
                    return new ApiResponse<>("success", "Patient found", patientOpt.get());
                } else {
                    return new ApiResponse<>("error", "Patient not found", null);
                }
            } else if (user.getRole() == User.Role.T_DOCTOR || user.getRole() == User.Role.B_DOCTOR) {
                Optional<Doctor> doctorOpt = doctorRepository.findByUser_UserId(userId);
                if (doctorOpt.isPresent()) {
                    return new ApiResponse<>("success", "Doctor found", doctorOpt.get());
                } else {
                    return new ApiResponse<>("error", "Doctor not found", null);
                }
            } else {
                return new ApiResponse<>("error", "User is neither a patient nor a doctor", null);
            }
        } else {
            return new ApiResponse<>("error", "User not found", null);
        }
    }
}
