package com.thd2020.pasmain.controller;

import com.thd2020.pasmain.dto.ApiResponse;
import com.thd2020.pasmain.dto.AuthenticationRequest;
import com.thd2020.pasmain.dto.JwtResponse;
import com.thd2020.pasmain.dto.RelatedIdsResponse;
import com.thd2020.pasmain.dto.UserRegistrationRequest;
import com.thd2020.pasmain.entity.*;
import com.thd2020.pasmain.service.*;
import com.thd2020.pasmain.util.JwtUtil;
import com.thd2020.pasmain.util.UtilFunctions;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpHeaders;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.AuthenticationException;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.oauth2.client.authentication.OAuth2AuthenticationToken;
import org.springframework.security.oauth2.core.user.OAuth2User;
import org.springframework.web.bind.annotation.*;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Optional;

@RestController
@RequestMapping("/api/v1/users")
public class UserController {

    @Autowired
    private UserService userService;

    @Autowired
    private DocInfoService docInfoService;

    @Autowired
    private PatientService patientService;

    @Autowired
    private PRInfoService prInfoService;

    @Autowired
    private AuthenticationManager authenticationManager;

    @Autowired
    private JwtUtil jwtUtil;

    @Autowired
    private UtilFunctions utilFunctions;

    @Autowired
    private ImagingService imagingService;

    // 用户注册
    @PostMapping("/register")
    @Operation(summary = "用户注册", description = "允许新用户注册账号，并关联或创建患者/医生信息。")
    public ApiResponse<User> registerUser(
            @Parameter(description = "用户注册信息", required = true)
            @RequestBody UserRegistrationRequest request) {
        // Validate required fields
        if (request.getPassId() == null || request.getPassId().isEmpty() ||
            request.getName() == null || request.getName().isEmpty()) {
            return new ApiResponse<>("error", "PassId and name are required", null);
        }

        try {
            User registeredUser = userService.registerUser(request);
            return new ApiResponse<>("success", "User registered successfully", registeredUser);
        } catch (Exception e) {
            return new ApiResponse<>("error", e.getMessage(), null);
        }
    }

    // 用户登录
    @PostMapping("/login")
    @Operation(summary = "用户登录", description = "允许用户使用用户名和密码登录。")
    public ApiResponse<JwtResponse> loginUser(
            @Parameter(description = "用户登录信息", required = true)
            @RequestBody AuthenticationRequest authenticationRequest) throws AuthenticationException {
        try {
            Authentication authentication = authenticationManager.authenticate(
                    new UsernamePasswordAuthenticationToken(authenticationRequest.getUsername(), authenticationRequest.getPassword()));
            User user = (User) authentication.getPrincipal();
            Long UserId = user.getUserId();
            String token = jwtUtil.generateToken(UserId, user.getUsername());
            User loginedUser = userService.loginUser(UserId).get();
            JwtResponse jwtResponse = new JwtResponse(token, loginedUser);
            return new ApiResponse<>("success", "Login successful", jwtResponse);
        } catch (AuthenticationException e) {
            return new ApiResponse<>("failed", "Username and Password not match", null);
        }
    }

    @PostMapping("/login/phone")
    @Operation(summary = "手机号登录", description = "允许用户使用手机号和密码登录。")
    public ApiResponse<JwtResponse> loginWithPhone(
            @Parameter(description = "手机号", required = true) @RequestParam String phone,
            @Parameter(description = "密码", required = true) @RequestParam String password) {
        Optional<User> userOpt = userService.loginWithPhone(phone, password);
        if (userOpt.isPresent()) {
            User user = userOpt.get();
            String token = jwtUtil.generateToken(user.getUserId(), user.getUsername());
            return new ApiResponse<>("success", "Login successful", new JwtResponse(token, user));
        }
        return new ApiResponse<>("error", "Invalid phone number or password", null);
    }

    @PostMapping("/login/email")
    @Operation(summary = "邮箱登录", description = "允许用户使用邮箱和密码登录。")
    public ApiResponse<JwtResponse> loginWithEmail(
            @Parameter(description = "邮箱", required = true) @RequestParam String email,
            @Parameter(description = "密码", required = true) @RequestParam String password) {
        Optional<User> userOpt = userService.loginWithEmail(email, password);
        if (userOpt.isPresent()) {
            User user = userOpt.get();
            String token = jwtUtil.generateToken(user.getUserId(), user.getUsername());
            return new ApiResponse<>("success", "Login successful", new JwtResponse(token, user));
        }
        return new ApiResponse<>("error", "Invalid email or password", null);
    }

    @PostMapping("/login/passid")
    @Operation(summary = "证件号登录", description = "允许用户使用证件号和密码登录。")
    public ApiResponse<JwtResponse> loginWithPassId(
            @Parameter(description = "证件号", required = true) @RequestParam String passId,
            @Parameter(description = "密码", required = true) @RequestParam String password) {
        Optional<User> userOpt = userService.loginWithPassId(passId, password);
        if (userOpt.isPresent()) {
            User user = userOpt.get();
            String token = jwtUtil.generateToken(user.getUserId(), user.getUsername());
            return new ApiResponse<>("success", "Login successful", new JwtResponse(token, user));
        }
        return new ApiResponse<>("error", "Invalid passId or password", null);
    }

    // Google OAuth2 登录成功回调
    @GetMapping("/login/oauth2/success")
    @Operation(summary = "Google OAuth2 登录成功回调", description = "处理Google OAuth2登录成功后的回调。")
    public ApiResponse<?> googleLoginSuccess() {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        if (authentication instanceof OAuth2AuthenticationToken oauth2Token) {
            OAuth2User oAuth2User = (OAuth2User) oauth2Token.getPrincipal();
            String email = oAuth2User.getAttribute("email");
            String name = oAuth2User.getAttribute("name");
            User user = userService.processOAuthPostLogin(email, name);
            String token = jwtUtil.generateToken(user.getUserId(), user.getUsername());
            JwtResponse jwtResponse = new JwtResponse(token, user);
            return new ApiResponse<>("success", "Login successful", jwtResponse);
        } else {
            return new ApiResponse<String>("error", "Authentication failed", null);
        }
    }

    // 发送验证码
    @PostMapping("/send-code")
    @Operation(summary = "发送验证码", description = "向用户的手机发送验证码，用于注册或登录。")
    public ApiResponse<?> sendVerificationCode(
            @Parameter(description = "手机号", required = true)
            @RequestParam String phone) throws IOException {
        userService.generateAndSendCode(phone);
        return new ApiResponse<>("success", "Verification code sent", null);
    }

    // 验证验证码
    @PostMapping("/verify-code")
    @Operation(summary = "验证验证码", description = "验证用户提供的验证码是否正确。")
    public ApiResponse<?> verifyCode(
            @Parameter(description = "手机号", required = true)
            @RequestParam String phone,
            @Parameter(description = "验证码", required = true)
            @RequestParam String code) throws IOException {
        if (userService.verifyCode(phone, code)) {
            User user = userService.findOrCreateUserByPhone(phone);
            String token = jwtUtil.generateToken(user.getUserId(), user.getUsername());
            return new ApiResponse<>("success", "Login successful", new JwtResponse(token, user));
        } else {
            return new ApiResponse<>("error", "Invalid verification code", null);
        }
    }

    // 获取用户信息
    @GetMapping("/{userId}")
    @Operation(summary = "获取用户信息", description = "允许管理员或用户本人获取用户的详细信息。")
    public ApiResponse<Optional<User>> getUserById(
            @Parameter(description = "要获取信息的用户ID", required = true)
            @PathVariable Long userId,
            @Parameter(description = "用于身份验证的JWT令牌，以\"Bearer \"开头", required = true)
            @RequestHeader("Authorization") String token) {
        String username = jwtUtil.extractUsername(token.substring(7));
        Optional<User> requestingUser = userService.getUserByUsername(username);
        if (requestingUser.isPresent() && (requestingUser.get().getRole() == User.Role.ADMIN || requestingUser.get().getUserId().equals(userId))) {
            Optional<User> user = userService.getUserById(userId);
            return new ApiResponse<>("success", "用户信息获取成功", user);
        } else {
            return new ApiResponse<>("failure", "未授权", Optional.empty());
        }
    }

    // 更新用户信息
    @PutMapping("/{userId}")
    @Operation(summary = "更新用户信息", description = "允许管理员或用户本人更新用户的详细信息。")
    public ApiResponse<JwtResponse> updateUser(
            @Parameter(description = "要更新信息的用户ID", required = true)
            @PathVariable Long userId,
            @Parameter(description = "更新后的用户信息", required = true)
            @RequestBody User updatedUser,
            @Parameter(description = "用于身份验证的JWT令牌", required = true)
            @RequestHeader("Authorization") String token) {
        String username = jwtUtil.extractUsername(token.substring(7));
        Optional<User> requestingUser = userService.getUserByUsername(username);
        if (requestingUser.isPresent() && (requestingUser.get().getRole() == User.Role.ADMIN || requestingUser.get().getUserId().equals(userId))) {
            Optional<User> user = userService.updateUser(userId, updatedUser);
            String newToken = jwtUtil.generateToken(user.orElseThrow().getUserId(), user.get().getUsername());
            JwtResponse jwtResponse = new JwtResponse(newToken, user.get());
            return new ApiResponse<>("success", "User updated successfully", jwtResponse);
        } else {
            return new ApiResponse<>("failure", "User not found", null);
        }
    }

    // 更改密码
    @PutMapping("/{userId}/change-password")
    @Operation(summary = "更改用户密码", description = "允许经过身份验证的用户更改自己的密码，或允许管理员更改任何用户的密码。")
    public ApiResponse<Boolean> changePassword(
            @Parameter(description = "要更改密码的用户ID", required = true)
            @PathVariable Long userId,
            @Parameter(description = "旧密码", required = true)
            @RequestBody String oldPassword,
            @Parameter(description = "新密码", required = true)
            @RequestBody String newPassword,
            @Parameter(description = "用于身份验证的JWT令牌", required = true)
            @RequestHeader("Authorization") String token) {
        String username = jwtUtil.extractUsername(token.substring(7));
        Optional<User> requestingUser = userService.getUserByUsername(username);
        if (requestingUser.isPresent() && (requestingUser.get().getRole() == User.Role.ADMIN || requestingUser.get().getUserId().equals(userId))) {
            boolean result = userService.changePassword(userId, oldPassword, newPassword);
            return new ApiResponse<>("success", "Password changed successfully", result);
        } else {
            return new ApiResponse<>("failure", "Unauthorized", false);
        }
    }

    // 重置密码
    @PostMapping("/reset-password")
    @Operation(summary = "重置密码", description = "允许用户通过电子邮件重置密码。")
    public ApiResponse<Boolean> resetPassword(
            @Parameter(description = "用户的电子邮件地址", required = true)
            @RequestParam String email,
            @Parameter(description = "新密码", required = true)
            @RequestParam String newPassword) {
        boolean result = userService.resetPassword(email, newPassword);
        return new ApiResponse<>("success", "Password reset successfully", result);
    }

    // 删除用户
    @DeleteMapping("/{userId}")
    @Operation(summary = "删除用户", description = "允许管理员删除用户。")
    public ApiResponse<Boolean> deleteUser(
            @Parameter(description = "要删除的用户ID", required = true)
            @PathVariable Long userId,
            @Parameter(description = "用于身份验证的JWT令牌", required = true)
            @RequestHeader("Authorization") String token) {
        String username = jwtUtil.extractUsername(token.substring(7));
        Optional<User> requestingUser = userService.getUserByUsername(username);

        if (requestingUser.isPresent() && requestingUser.get().getRole() == User.Role.ADMIN) {
            boolean result = userService.deleteUser(userId);
            return new ApiResponse<>("success", "User deleted successfully", result);
        } else {
            return new ApiResponse<>("failure", "Unauthorized", false);
        }
    }

    // 用户登出
    @PostMapping("/logout")
    @Operation(summary = "用户登出", description = "允许用户登出。")
    public ApiResponse<Void> logoutUser(
            @Parameter(description = "用于身份验证的JWT令牌", required = true)
            @RequestHeader(HttpHeaders.AUTHORIZATION) String authorizationHeader) {
        if (authorizationHeader != null && authorizationHeader.startsWith("Bearer ")) {
            String token = authorizationHeader.substring(7); // Remove "Bearer " prefix
            userService.logoutUser(token);
        }
        return new ApiResponse<>("success", "User logged out successfully", null);
    }


    // 获取所有用户列表（管理员权限）
    @GetMapping
    @Operation(summary = "获取所有用户列表", description = "允许管理员获取所有用户的列表。")
    public ApiResponse<List<User>> getAllUsers(
            @Parameter(description = "用于身份验证的JWT令牌", required = true)
            @RequestHeader("Authorization") String token) {
        String username = jwtUtil.extractUsername(token.substring(7));
        Optional<User> requestingUser = userService.getUserByUsername(username);
        if (requestingUser.isPresent() && requestingUser.get().getRole() == User.Role.ADMIN) {
            List<User> users = userService.getAllUsers();
            return new ApiResponse<>("success", "Users fetched successfully", users);
        } else {
            return new ApiResponse<>("failure", "Unauthorized", null);
        }
    }

    // 分配用户角色（管理员权限）
    @PutMapping("/{userId}/assign-role")
    @Operation(summary = "分配用户角色", description = "允许管理员分配用户角色。")
    public ApiResponse<Optional<User>> assignRole(
            @Parameter(description = "要分配角色的用户ID", required = true)
            @PathVariable Long userId,
            @Parameter(description = "要分配的角色", required = true)
            @RequestParam String role,
            @Parameter(description = "用于身份验证的JWT令牌", required = true)
            @RequestHeader("Authorization") String token) {
        String username = jwtUtil.extractUsername(token.substring(7));
        Optional<User> requestingUser = userService.getUserByUsername(username);
        if (requestingUser.isPresent() && requestingUser.get().getRole() == User.Role.ADMIN) {
            User.Role userRole = User.Role.valueOf(role.toUpperCase());
            Optional<User> user = userService.assignRole(userId, userRole);
            return new ApiResponse<>("success", "Role assigned successfully", user);
        } else {
            return new ApiResponse<>("failure", "Unauthorized", Optional.empty());
        }
    }

    @GetMapping("/find-by-username")
    @Operation(summary = "通过用户名查询用户ID", description = "允许管理员,医生通过用户名查询所有对应名字的用户ID")
    public ApiResponse<Long> getUserIdsByUsername(
            @Parameter(description = "JWT token用于身份验证", required = true) @RequestHeader("Authorization") String token,
            @Parameter(description = "用户名", required = true) @RequestParam String username) {

        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token)) {
            Long userId = userService.findUserIdsByUsername(username);
            if (userId != null) {
                return new ApiResponse<>("success", "User IDs fetched successfully", userId);
            }
            else {
                return new ApiResponse<>("failure", "No such user", null);
            }
        } else {
            return new ApiResponse<>("failure", "Unauthorized", null);
        }
    }

    @GetMapping("/related-ids/{userId}")
    @Operation(summary = "通过用户ID查询相关记录ID", description = "允许管理员，医生，病人本人通过用户ID查询Patient的patientId，MedicalRecord的RecordId，SurgeryAndBloodTest的RecordId，以及UltrasoundScore的ScoreId，以及ImagingRecord的recordId")
    public ApiResponse<?> getRelatedIdsByUserId(
            @Parameter(description = "JWT token用于身份验证", required = true) @RequestHeader("Authorization") String token,
            @Parameter(description = "用户ID", required = true) @PathVariable Long userId) {

        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token) || utilFunctions.isMatch(token, userId)) {
            Optional<Patient> patient = patientService.findPatientByUserId(userId);
            if (patient.isPresent()) {
                String patientId = patient.get().getPatientId();
                List<MedicalRecord> medicalRecordIds = prInfoService.findMedicalRecordIdsByPatientId(patientId);
                List<SurgeryAndBloodTest> surgeryAndBloodTestIds = prInfoService.findSBRecordIdsByPatientId(patientId);
                List<UltrasoundScore> ultrasoundScoreIds = prInfoService.findScoreIdsByPatientId(patientId);
                List<ImagingRecord> imagingRecordIds = imagingService.findImagingRecordIds(patientId);
                return new ApiResponse<>("success", "Related IDs fetched successfully", Map.of(
                        "patientId", patientId,
                        "MedicalRecordIds", medicalRecordIds,
                        "surgeryAndBloodTestIds", surgeryAndBloodTestIds,
                        "ultrasoundScoreIds", ultrasoundScoreIds,
                        "imagingRecordIds", imagingRecordIds));
            }
            else {
                return new ApiResponse<>("failure", "No such patient corresponding to this user", null);
            }
        } else {
            return new ApiResponse<>("failure", "Unauthorized", null);
        }
    }

    @Operation(summary = "查询用户的ID", description = "根据用户ID查询对应的患者ID或医生ID")
    @GetMapping("/{userId}/details")
    public ApiResponse<?> getUserDetails(
            @Parameter(description = "用户ID", required = true) @PathVariable Long userId,
            @RequestHeader("Authorization") String token) {
        if (utilFunctions.isAdmin(token) || utilFunctions.isDoctor(token) || utilFunctions.isMatch(token, userId)) {
            return userService.getUserDetails(userId);
        } else {
            return new ApiResponse<>("error", "Unauthorized", null);
        }
    }

    @Operation(summary = "查询token状态", description = "查询token过期状态")
    @GetMapping("/expired")
    public ApiResponse<?> getUserDetails(
            @Parameter(description = "要查询的token", required = true)
            @RequestParam String token) {
        try {
            return new ApiResponse<>("success", "token available", jwtUtil.extractAllClaims(token));
        }
        catch (Exception e) {
            return new ApiResponse<>("failure", "token unavailable", e);
        }
    }
}