package com.thd2020.pasmain;

import org.opencv.core.Core;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import java.net.URL;

@SpringBootApplication
public class PasMainApplication {

    static {
        try {
            URL url = ClassLoader.getSystemResource("lib/libopencv_java4100.so");
            System.load(url.getPath());
        }
        catch (UnsatisfiedLinkError ignore) { }
    }

    public static void main(String[] args) {
        SpringApplication.run(PasMainApplication.class, args);
    }

}
