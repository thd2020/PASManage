package com.thd2020.pasapi.entity;

import javax.persistence.*;
import lombok.Data;

@Entity
@Data
public class ApiDoc {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String title;
    private String description;
    private String endpoint;
    private String method;
    private String params;
    private String types;
    private String expectedReturns;
    private String examples;
}
