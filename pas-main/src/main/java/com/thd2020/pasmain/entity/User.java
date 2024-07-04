package com.thd2020.pasmain.entity;

import jakarta.persistence.*;
import lombok.Data;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.userdetails.UserDetails;

import java.time.LocalDateTime;
import java.util.Collection;
import java.util.List;

@Entity
@Data
public class User implements UserDetails {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long userId;

    @Column(nullable = false, length = 50)
    private String username;

    @Column(nullable = false, length = 255)
    private String password;

    @Column(nullable = false, length = 100)
    private String email;

    @Column(nullable = false, length = 20)
    private String phone;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private Role role;

    @Column(nullable = false)
    private LocalDateTime createdAt;

    private LocalDateTime lastLogin;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private Status status;

    @Override
    public Collection<? extends GrantedAuthority> getAuthorities() {
        return new java.util.ArrayList<>();
    }

    @Enumerated(EnumType.STRING)
    private Provider provider;

    public enum Role {
        T_DOCTOR,
        B_DOCTOR,
        PATIENT,
        ADMIN
    }

    public enum Status {
        ACTIVE,
        INACTIVE,
        BANNED
    }

    public enum Provider {
        LOCAL, GOOGLE
    }
}
