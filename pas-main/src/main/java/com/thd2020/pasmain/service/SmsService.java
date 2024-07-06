package com.thd2020.pasmain.service;

import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.IOException;

@Service
public class SmsService {

    @Value("${releans.api.key}")
    private String apiKey;

    public void sendSms(String phone, String message) throws IOException {
        String url = "https://api.releans.com/v2/message";
        CloseableHttpClient client = HttpClients.createDefault();
        HttpPost httpPost = new HttpPost(url);

        String payload = String.format("sender=YourSenderName&mobile=%s&content=%s", phone, message);
        StringEntity entity = new StringEntity(payload);

        httpPost.setEntity(entity);
        httpPost.setHeader("Authorization", "Bearer " + apiKey);
        httpPost.setHeader("Content-Type", "application/x-www-form-urlencoded");

        CloseableHttpResponse response = client.execute(httpPost);
        System.out.println(EntityUtils.toString(response.getEntity()));
        client.close();
    }
}
