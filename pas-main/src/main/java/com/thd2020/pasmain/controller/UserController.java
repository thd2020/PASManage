package com.thd2020.pasmain.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import com.thd2020.pasmain.entity.User;
import com.thd2020.pasmain.service.UserService;
import java.util.List;
import java.util.Optional;

@RestController
@RequestMapping("/api/v1/users")
public class UserController {
    @Autowired
    private UserService userService;

    @PostMapping("/register")
    public User registerUser(@RequestBody User user) {
        return userService.registerUser(user);
    }

    @PostMapping("/login")
    public Optional<User> loginUser(@RequestParam String username, @RequestParam String password) {
        return userService.loginUser(username, password);
    }

    @GetMapping("/{userId}")
    public Optional<User> getUserById(@PathVariable Long userId) {
        return userService.getUserById(userId);
    }

    @PutMapping("/{userId}")
    public Optional<User> updateUser(@PathVariable Long userId, @RequestBody User updatedUser) {
        return userService.updateUser(userId, updatedUser);
    }

    @PutMapping("/{userId}/change-password")
    public boolean changePassword(@PathVariable Long userId, @RequestParam String oldPassword, @RequestParam String newPassword) {
        return userService.changePassword(userId, oldPassword, newPassword);
    }

    @PostMapping("/reset-password")
    public boolean resetPassword(@RequestParam String email, @RequestParam String newPassword) {
        return userService.resetPassword(email, newPassword);
    }

    @DeleteMapping("/{userId}")
    public boolean deleteUser(@PathVariable Long userId) {
        return userService.deleteUser(userId);
    }

    @PostMapping("/logout")
    public void logoutUser() {
        userService.logoutUser();
    }

    @GetMapping
    public List<User> getAllUsers() {
        return userService.getAllUsers();
    }

    @PutMapping("/{userId}/assign-role")
    public Optional<User> assignRole(@PathVariable Long userId, @RequestParam String role) {
        User.Role userRole = User.Role.valueOf(role.toUpperCase());
        return userService.assignRole(userId, userRole);
    }
}
