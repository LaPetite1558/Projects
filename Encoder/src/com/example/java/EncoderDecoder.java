package com.example.java;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.stream.Collectors;

public class EncoderDecoder {
    private static final char defaultOffset = 'B';
    private final CustomIterator customIterator = new CustomIterator();

    public String encode(String plainText) {
        List<Character> result = new ArrayList<>();
        customIterator.setCurrent(0);
        customIterator.setEncode(true);
        customIterator.setOffsetChar(defaultOffset);
        customIterator.setChars(plainText.toUpperCase(Locale.ROOT).toCharArray());
        while (customIterator.hasNext()) {
            result.add(customIterator.next());
        }
        return String.format("%s%s", defaultOffset, result.stream().map(String::valueOf).collect(Collectors.joining()));
    }

    public String decode(String encodedText) {
        List<Character> result = new ArrayList<>();
        customIterator.setCurrent(0);
        customIterator.setEncode(false);
        customIterator.setOffsetChar(encodedText.charAt(0));
        customIterator.setChars(encodedText.substring(1).toCharArray());
        while (customIterator.hasNext()) {
            result.add(customIterator.next());
        }
        return result.stream().map(String::valueOf).collect(Collectors.joining());
    }
}
