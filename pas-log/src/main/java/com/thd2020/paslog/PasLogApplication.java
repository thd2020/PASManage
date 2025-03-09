package com.thd2020.paslog;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.autoconfigure.domain.EntityScan;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.data.jpa.repository.config.EnableJpaRepositories;

@SpringBootApplication
@EntityScan(basePackages = "com.thd2020.paslog.entity")
@EnableJpaRepositories("com.thd2020.paslog.entity")
@ComponentScan(basePackages = { "com.thd2020.paslog.entity" })
public class PasLogApplication {

	public static void main(String[] args) {
		SpringApplication.run(PasLogApplication.class, args);
	}

}
