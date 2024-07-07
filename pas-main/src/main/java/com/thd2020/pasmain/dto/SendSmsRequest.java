package com.thd2020.pasmain.dto;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class SendSmsRequest {
    // Getters and setters
    private String to;
    private String signature;
}
