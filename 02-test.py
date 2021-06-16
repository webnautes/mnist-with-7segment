import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model


import time
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)

digitBitmap = { 0: 0b00111111, 1: 0b00000110, 2: 0b01011011, 3: 0b01001111, 4: 0b01100110, 5: 0b01101101, 6: 0b01111101, 7: 0b00000111, 8: 0b01111111, 9: 0b01100111 }
masks = { 'a': 0b00000001, 'b': 0b00000010, 'c': 0b00000100, 'd': 0b00001000, 'e': 0b00010000, 'f': 0b00100000, 'g': 0b01000000 }

pins = { 'a': 17, 'b': 27, 'c': 22, 'd': 10, 'e': 9, 'f': 11, 'g': 0}

def renderChar(c):
    val = digitBitmap[c]

    GPIO.output(list(pins.values()), GPIO.HIGH)

    for k,v in masks.items():
        if val&v == v:
            GPIO.output(pins[k], GPIO.LOW)



GPIO.setup(list(pins.values()), GPIO.OUT)
GPIO.output(list(pins.values()), GPIO.HIGH)


import os
model = load_model(os.path.join("./","model.h5"))


while True:

    try:
    



      for i in range(10):
  
          img_color = cv.imread(str(i) + '.png', cv.IMREAD_COLOR)
          img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
  
  
          ret,img_binary = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
  
          kernel = cv.getStructuringElement( cv.MORPH_RECT, ( 5, 5 ) )
          img_binary = cv.morphologyEx(img_binary, cv. MORPH_CLOSE, kernel)
  
  
          contours, hierarchy = cv.findContours(img_binary, cv.RETR_EXTERNAL, 
                                  cv.CHAIN_APPROX_SIMPLE)
  
          max_contour = -1
          max_area = -1
  
          for contour in contours:
              area = cv.contourArea(contour)
  
              if area > max_area:
                  max_area = area
                  max_contour = contour
  
          x, y, w, h = cv.boundingRect(max_contour)
  
          img = img_binary[y:y+h, x:x+w]
          
  
  
          desired_size = max(w, h) + 100
          new_width = w
          new_height = h
  
          delta_w = desired_size - new_width
          delta_h = desired_size - new_height
          top, bottom = delta_h//2, delta_h-(delta_h//2)
          left, right = delta_w//2, delta_w-(delta_w//2)
  
          new_img = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, value=[0, 0, 0])
  
  
          kernel = np.ones((5, 5), np.uint8)
          img_digit = cv.morphologyEx(new_img, cv.MORPH_DILATE, kernel)
  
  
          img_digit = cv.resize(img_digit, (28, 28), interpolation=cv.INTER_AREA)
  
          img_digit = img_digit / 255.0
  
          img_input = img_digit.reshape(1, 28, 28, 1)
          predictions = model.predict(img_input)
  
  
          number = np.argmax(predictions)
          print(number)
  
          cv.rectangle(img_color, (x, y), (x+w, y+h), (255, 255, 0), 2)    
  
          renderChar(number)
  
          cv.imshow('result', img_color)
          cv.waitKey(1000)

    except KeyboardInterrupt:
        print("End")
        break


GPIO.cleanup()
