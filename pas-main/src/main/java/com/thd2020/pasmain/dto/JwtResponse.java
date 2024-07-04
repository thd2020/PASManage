package com.thd2020.pasmain.dto;

import com.thd2020.pasmain.entity.User;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class JwtResponse {
    private String token;
    private User user;

    public JwtResponse(String token, User user) {
        this.token = token;
        this.user = user;
    }

}
