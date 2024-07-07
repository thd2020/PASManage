package com.thd2020.pasmain.dto;

import lombok.Getter;
import lombok.Setter;

@Setter
@Getter
public class VerifySmsRequest {
    // Getters and setters
    private String to;
    private String code;
}