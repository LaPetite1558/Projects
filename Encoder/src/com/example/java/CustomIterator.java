package com.example.java;

import java.util.Iterator;
import java.util.NoSuchElementException;

public class CustomIterator implements Iterator<Character> {
    public static final String validChars= "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789()*+,-./";
    public static final int validCharLen = validChars.length();
    private int current;
    private char offsetChar;
    private boolean isEncode;
    private char[] chars;

    public CustomIterator(char[] chars, char offsetChar, boolean isEncode) {
        this.chars = chars;
        this.offsetChar = offsetChar;
        this.isEncode = isEncode;
        this.current = 0;
    }
    public CustomIterator() {
        this.current = 0;
    }

    @Override
    public boolean hasNext() {
        return current < chars.length;
    }

    public void setCurrent(int current) {
        this.current = current;
    }

    @Override
    public Character next() {
        Character out;
        if (!hasNext()) {
            throw new NoSuchElementException("no more characters in string");
        }
        if (isEncode) {
            out = encodeChar(chars[current++]);
        } else out = decodeChar(chars[current++]);
        return out;
    }

    private Character encodeChar(char c) {
        char out = c;
        if (validChars.indexOf(c) >= 0) {
            int newPos = (validChars.indexOf(c) - validChars.indexOf(offsetChar)) % validCharLen;
            if (newPos < 0) {
                newPos += validCharLen;
            } else if (newPos == validCharLen) {
                newPos -= validCharLen;
            }
            out = validChars.charAt(newPos);
        }
        return out;
    }

    private Character decodeChar(char c) {
        char out = c;
        if (validChars.indexOf(c) >= 0) {
            out = validChars.charAt((validChars.indexOf(c) + validChars.indexOf(offsetChar)) % validCharLen);
        }
        return out;
    }

    public void setOffsetChar(char offsetChar) {
        this.offsetChar = offsetChar;
    }

    public void setEncode(boolean encode) {
        isEncode = encode;
    }

    public void setChars(char[] chars) {
        this.chars = chars;
    }
}
