package com.example.java;

public class Main {
    public static void main(String[] args) {
        String plainText = "HELLO WORLD";
        EncoderDecoder encoderDecoder = new EncoderDecoder();

        String encodedText = encoderDecoder.encode(plainText);
        String decodedText = encoderDecoder.decode(encodedText);
        System.out.println(encodedText);
        System.out.println(decodedText);
    }
}
