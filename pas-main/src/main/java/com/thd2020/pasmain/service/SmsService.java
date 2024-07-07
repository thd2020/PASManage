package com.thd2020.pasmain.service;

import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.ContentType;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.IOException;

@Service
public class SmsService {

    @Value("${unimtx.api.key}")
    private String apiKey;

    @Value("${unimtx.send-otp.url}")
    private String sendOtpUrl;

    @Value("${unimtx.verify-otp.url}")
    private String verifyOtpUrl;

    public void sendSms(String phone) throws IOException {
        CloseableHttpClient client = HttpClients.createDefault();
        String url = sendOtpUrl + "&accessKeyId=" + apiKey;

        HttpPost httpPost = new HttpPost(url);
        String payload = String.format("{\"to\":\"%s\",\"signature\":\"合一矩阵\"}", phone);
        StringEntity entity = new StringEntity(payload, "application/json", "UTF-8");

        httpPost.setEntity(entity);
        httpPost.setHeader("Content-Type", "application/json");

        CloseableHttpResponse response = client.execute(httpPost);
        System.out.println(EntityUtils.toString(response.getEntity()));
        client.close();
    }

    public boolean verifySms(String phone, String code) throws IOException {
        CloseableHttpClient client = HttpClients.createDefault();
        String url = verifyOtpUrl + "&accessKeyId=" + apiKey;

        HttpPost httpPost = new HttpPost(url);
        String payload = String.format("{\"to\":\"%s\",\"code\":\"%s\"}", phone, code);
        StringEntity entity = new StringEntity(payload, "application/json", "UTF-8");

        httpPost.setEntity(entity);
        httpPost.setHeader("Content-Type", "application/json");

        CloseableHttpResponse response = client.execute(httpPost);
        String responseBody = EntityUtils.toString(response.getEntity());
        client.close();

        // 解析响应
        return responseBody.contains("\"valid\":true");
    }
}
