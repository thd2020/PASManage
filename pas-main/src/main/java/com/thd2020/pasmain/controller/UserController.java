package com.thd2020.pasmain.controller;

import com.thd2020.pasmain.dto.ApiResponse;
import com.thd2020.pasmain.dto.AuthenticationRequest;
import com.thd2020.pasmain.dto.JwtResponse;
import com.thd2020.pasmain.entity.User;
import com.thd2020.pasmain.entity.Patient;
import com.thd2020.pasmain.entity.Doctor;
import com.thd2020.pasmain.service.InfoService;
import com.thd2020.pasmain.service.UserService;
import com.thd2020.pasmain.util.JwtUtil;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.AuthenticationException;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Optional;

@RestController
@RequestMapping("/api/v1/users")
public class UserController {

    @Autowired
    private UserService userService;

    @Autowired
    private InfoService infoService;

    @Autowired
    private AuthenticationManager authenticationManager;

    @Autowired
    private JwtUtil jwtUtil;

    // 用户注册
    @PostMapping("/register")
    public ApiResponse<User> registerUser(@RequestBody User user) {
        User registeredUser = userService.registerUser(user);

        // 根据角色将用户关联到病人或医生
        if (user.getRole() == User.Role.PATIENT) {
            Patient patient = new Patient();
            patient.setUser(registeredUser);
            infoService.savePatient(patient);
        } else if (user.getRole() == User.Role.T_DOCTOR || user.getRole() == User.Role.B_DOCTOR) {
            Doctor doctor = new Doctor();
            doctor.setUser(registeredUser);
            infoService.saveDoctor(doctor);
        }

        return new ApiResponse<>("success", "User registered successfully", registeredUser);
    }

    // 用户登录
    @PostMapping("/login")
    public ApiResponse<JwtResponse> loginUser(@RequestBody AuthenticationRequest authenticationRequest) throws AuthenticationException {
        Authentication authentication = authenticationManager.authenticate(
                new UsernamePasswordAuthenticationToken(authenticationRequest.getUsername(), authenticationRequest.getPassword()));
        User user = (User) authentication.getPrincipal();
        String token = jwtUtil.generateToken(user.getUsername());
        JwtResponse jwtResponse = new JwtResponse(token);
        return new ApiResponse<>("success", "Login successful", jwtResponse);
    }

    // 获取用户信息
    @GetMapping("/{userId}")
    public ApiResponse<Optional<User>> getUserById(@PathVariable Long userId, @RequestHeader("Authorization") String token) {
        String username = jwtUtil.extractUsername(token.substring(7));
        Optional<User> requestingUser = userService.getUserByUsername(username);

        if (requestingUser.isPresent() && (requestingUser.get().getRole() == User.Role.ADMIN || requestingUser.get().getUserId().equals(userId))) {
            Optional<User> user = userService.getUserById(userId);
            return new ApiResponse<>("success", "User fetched successfully", user);
        } else {
            return new ApiResponse<>("failure", "Unauthorized", Optional.empty());
        }
    }

    // 更新用户信息
    @PutMapping("/{userId}")
    public ApiResponse<Optional<User>> updateUser(@PathVariable Long userId, @RequestBody User updatedUser, @RequestHeader("Authorization") String token) {
        String username = jwtUtil.extractUsername(token.substring(7));
        Optional<User> requestingUser = userService.getUserByUsername(username);

        if (requestingUser.isPresent() && (requestingUser.get().getRole() == User.Role.ADMIN || requestingUser.get().getUserId().equals(userId))) {
            Optional<User> user = userService.updateUser(userId, updatedUser);
            return new ApiResponse<>("success", "User updated successfully", user);
        } else {
            return new ApiResponse<>("failure", "Unauthorized", Optional.empty());
        }
    }

    // 更改密码
    @PutMapping("/{userId}/change-password")
    public ApiResponse<Boolean> changePassword(@PathVariable Long userId, @RequestParam String oldPassword, @RequestParam String newPassword, @RequestHeader("Authorization") String token) {
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
    public ApiResponse<Boolean> resetPassword(@RequestParam String email, @RequestParam String newPassword) {
        boolean result = userService.resetPassword(email, newPassword);
        return new ApiResponse<>("success", "Password reset successfully", result);
    }

    // 删除用户
    @DeleteMapping("/{userId}")
    public ApiResponse<Boolean> deleteUser(@PathVariable Long userId, @RequestHeader("Authorization") String token) {
        String username = jwtUtil.extractUsername(token.substring(7));
        Optional<User> requestingUser = userService.getUserByUsername(username);

        if (requestingUser.isPresent() && requestingUser.get().getRole() == User.Role.ADMIN) {
            boolean result = userService.deleteUser(userId);
            return new ApiResponse<>("success", "User deleted successfully", result);
        } else {
            return new ApiResponse<>("failure", "Unauthorized", false);
        }
    }

    // 用户注销
    @PostMapping("/logout")
    public ApiResponse<Void> logoutUser(@RequestHeader("Authorization") String token) {
        String username = jwtUtil.extractUsername(token.substring(7));
        userService.logoutUser();
        return new ApiResponse<>("success", "User logged out successfully", null);
    }

    // 获取所有用户列表（管理员权限）
    @GetMapping
    public ApiResponse<List<User>> getAllUsers(@RequestHeader("Authorization") String token) {
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
    public ApiResponse<Optional<User>> assignRole(@PathVariable Long userId, @RequestParam String role, @RequestHeader("Authorization") String token) {
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
}