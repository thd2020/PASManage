package com.thd2020.pasmain.service;

import com.thd2020.pasmain.dto.OAuthRequest;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import com.thd2020.pasmain.entity.User;
import com.thd2020.pasmain.repository.UserRepository;

import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;

@Service
public class UserService {
    @Autowired
    private UserRepository userRepository;

    @Autowired
    private PasswordEncoder passwordEncoder;

    @Autowired
    private TokenBlacklistService tokenBlacklistService;

    @Autowired
    private SmsService smsService;

    private final Map<String, String> verificationCodes = new HashMap<>();

    // 用户注册
    public User registerUser(User user) {
        user.setPassword(passwordEncoder.encode(user.getPassword()));
        user.setCreatedAt(LocalDateTime.now());
        user.setStatus(User.Status.ACTIVE); // 默认设置为ACTIVE
        return userRepository.save(user);
    }

    public User processOAuthPostLogin(String email, String userName) {
        Optional<User> existUser = userRepository.findByEmail(email);
        if (existUser.isEmpty()) {
            User newUser = new User();
            newUser.setUsername(userName);
            newUser.setEmail(email);
            newUser.setProvider(User.Provider.GOOGLE);
            newUser.setCreatedAt(LocalDateTime.now());
            newUser.setStatus(User.Status.ACTIVE); // 默认设置为ACTIVE
            String rawPassword = String.format("%s:%s:%s", userName, email, LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMddHHmmss")));
            newUser.setPassword(passwordEncoder.encode(rawPassword));
            newUser.setRole(User.Role.PATIENT);
            return userRepository.save(newUser);
        }
        else{
            return existUser.get();
        }
    }


    // 用户登录
    public Optional<User> loginUser(String username, String password) {
        Optional<User> userOptional = userRepository.findByUsername(username);
        if (userOptional.isPresent()) {
            User user = userOptional.get();
            user.setLastLogin(LocalDateTime.now());
            user = userRepository.save(user);
            if (passwordEncoder.matches(password, user.getPassword())) {
                return Optional.of(user);
            }
        }
        return Optional.empty();
    }

    public String generateAndSendCode(String phone) {
        String code = String.valueOf(new Random().nextInt(900000) + 100000);
        verificationCodes.put(phone, code);
        try {
            smsService.sendSms(phone, "Your verification code is " + code);
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Failed to send verification code");
        }
        return code;
    }

    public boolean verifyCode(String phone, String code) {
        return code.equals(verificationCodes.get(phone));
    }

    public User findOrCreateUserByPhone(String phone) {
        if (!userRepository.findByPhone(phone).isPresent()) {
            User user = new User();
            user.setPhone(phone);
            user.setUsername(phone);
            user.setCreatedAt(LocalDateTime.now());
            user.setStatus(User.Status.ACTIVE); // 默认设置为ACTIVE
            String rawPassword = String.format("%s:%s:%s", phone, LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMddHHmmss")));
            user.setPassword(passwordEncoder.encode(rawPassword));
            user.setRole(User.Role.PATIENT);
            return userRepository.save(user);
        }
        else{
            return userRepository.findByPhone(phone).get();
        }
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
}
