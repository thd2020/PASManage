package com.thd2020.pasmain.config;

import org.springframework.amqp.core.Binding;
import org.springframework.amqp.core.BindingBuilder;
import org.springframework.amqp.core.Queue;
import org.springframework.amqp.core.TopicExchange;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class RabbitMQConfig {
    @Bean
    public Queue queue() {
        return new Queue("referralQueue", false);  // 创建一个名为 "referralQueue" 的队列
    }

    @Bean
    public Queue referralQueue() {
        // Ensure the queue is created on the RabbitMQ server before sending any messages to it.
        return new Queue("hospital.127.0.0.1.queue", true); // Durable queue
    }

    @Bean
    public TopicExchange referralExchange() {
        return new TopicExchange("referral.exchange");
    }

    public Queue createReferralQueue(String hospitalIp) {
        return new Queue("hospital." + hospitalIp + ".queue", true);
    }

    @Bean
    public Binding referralBinding(Queue referralQueue, TopicExchange referralExchange) {
        return BindingBuilder.bind(referralQueue).to(referralExchange).with("referral.#");
    }

}