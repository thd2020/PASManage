package com.thd2020.pasmain.dto;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class OAuthRequest {
    private String email;
    private String phone;
    private String provider; // 如 "google", "facebook"
    private String oauthToken; // OAuth 提供的令牌
    // Getters and setters
}