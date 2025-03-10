package com.thd2020.pasmain.dto;

import com.thd2020.pasmain.entity.User;
import lombok.Data;

@Data
public class UserRegistrationRequest {
    private String username;
    private String password;
    private String email;
    private String phone;
    private User.Role role;
    private String name;  // True name
    private String passId;
}
