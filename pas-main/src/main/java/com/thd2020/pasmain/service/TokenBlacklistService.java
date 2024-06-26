package com.thd2020.pasmain.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Service;

import java.util.concurrent.TimeUnit;

@Service
public class TokenBlacklistService {

    @Autowired
    private StringRedisTemplate redisTemplate;

    private static final long TOKEN_EXPIRATION_TIME = 86400; // token有效期，单位为秒

    public void blacklistToken(String token) {
        redisTemplate.opsForValue().set(token, "blacklisted", TOKEN_EXPIRATION_TIME, TimeUnit.SECONDS);
    }

    public boolean isTokenBlacklisted(String token) {
        return Boolean.TRUE.equals(redisTemplate.hasKey(token));
    }
}
