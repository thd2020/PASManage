package com.thd2020.pasmain.dto;

import com.thd2020.pasmain.entity.User;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@AllArgsConstructor
public class JwtResponse {
    private String token;
    private User user;
}
