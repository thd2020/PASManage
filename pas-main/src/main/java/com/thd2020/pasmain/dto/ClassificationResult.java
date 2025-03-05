package com.thd2020.pasmain.dto;

import lombok.Data;
import java.util.Map;

@Data
public class ClassificationResult {
    private Map<String, Double> probabilities;
    private String predictedType;
}
