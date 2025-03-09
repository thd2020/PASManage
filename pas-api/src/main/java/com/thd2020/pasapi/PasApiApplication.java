package com.thd2020.pasapi;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.autoconfigure.domain.EntityScan;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.data.jpa.repository.config.EnableJpaRepositories;

@SpringBootApplication
@EntityScan(basePackages = "com.thd2020.pasapi.entity")
@EnableJpaRepositories("com.thd2020.pasapi.entity")
@ComponentScan(basePackages = { "com.thd2020.pasapi.entity" })
public class PasApiApplication {

	public static void main(String[] args) {
		SpringApplication.run(PasApiApplication.class, args);
	}

}
