package com.thd2020.pasmain.controller;

import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketSession;
import org.springframework.web.socket.handler.TextWebSocketHandler;

public class MyWebSocketHandler extends TextWebSocketHandler {

    @Override
    protected void handleTextMessage(WebSocketSession session, TextMessage message) throws Exception {
        // Here, we can send a response message to the client
        String clientMessage = message.getPayload();
        session.sendMessage(new TextMessage("Server received: " + clientMessage));
    }
}