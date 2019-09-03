#!/usr/bin/python
#coding: utf-8

# file: detect7segments.py
# brief: This file contains a simple detection numerical of 7-segment display
# author: Petrucio Ricardo Tavares de Medeiros
# date: 21/08/2019

import sys
import cv2
import numpy as np

# define the dictionary of digit segments so we can identify each digit
DIGITS_LOOKUP = {
    (1, 1, 1, 0, 1, 1, 1): 0,
    (0, 0, 1, 0, 0, 1, 0): 1,
    (1, 0, 1, 1, 1, 0, 1): 2,
    (1, 0, 1, 1, 0, 1, 1): 3,
    (0, 1, 1, 1, 0, 1, 0): 4,
    (1, 1, 0, 1, 0, 1, 1): 5,
    (1, 1, 0, 1, 1, 1, 1): 6,
    (1, 0, 1, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9
}

def filterImage( file ):
    gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
    gray = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    edged = cv2.Canny(gray, 100, 200)
    kernel = np.ones((3,3),np.uint8)
    #edged = cv2.dilate(edged, kernel, iterations = 2)
    return edged


def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return cnts, boundingBoxes

def delimitRegion( file, digit ):
    height, width = file.shape[:2]
    xmin = width; xmax = 0
    ymin = height; ymax = 0
    for c in digit:
        (x, y, w, h) = cv2.boundingRect(c) # Compute the bounding box of the contour
        if ( xmin > x ):
            xmin = x
        if ( xmax < x+w ):
            xmax = x+w
        if ( ymin > y ):
            ymin = y
        if ( ymax < y+h ):
            ymax = y+h
    
    return xmin, ymin, xmax, ymax
 

def identifyNumber( file, digit, xmin, ymin, xmax, ymax ):
    on = [0] * 7 # display 7 segments - removing 1, because it has another configuration
    rxmax = xmax - xmin; rymax = ymax - ymin; perc = 0.9
    #if ( len( digit )  == 1 ) or ( len( digit ) == 2 ): return 1
    cut1_4xmax = ((1.0/4.0) * rxmax) * perc; cut1_4ymax = ((1.0/4.0) * rymax) * perc
    cut2_4xmax = ((2.0/4.0) * rxmax) * perc; cut2_4ymax = ((2.0/4.0) * rymax) * perc
    cut3_4xmax = ((3.0/4.0) * rxmax) * perc; cut3_4ymax = ((3.0/4.0) * rymax) * perc
    for c in digit:
        #(x, y, w, h) = cv2.boundingRect(c) # Compute the bounding box of the contour
        # (x,y) upper left point
        # (x+w, y+h) lower right point
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        
        #cv2.rectangle(file, (xmax - int(cut1_4xmax), ymin + int(cut2_4ymax)), (xmax, ymax), (255, 255, 255), 1)
        #cv2.rectangle(file, (xmin, ymin + int(cut1_4xmax)), (xmin + int(cut2_4xmax), ymin + int(cut2_4ymax)), (255, 255, 255), 1)
        #cv2.imshow( "detected", file )
        #cv2.waitKey( 0 )
                    
        if ( ( cX > xmin ) and ( cX < xmax ) and
             ( cY > ymin ) and ( cY < ymin + int(cut1_4ymax) ) ): on[0] = 1        
        if ( ( cX > xmin ) and ( cX < xmin + int(cut1_4xmax) ) and
             ( cY > ymin ) and ( cY < ymin + int(cut2_4ymax) ) ): on[1] = 1
        if ( ( cX > xmin + int(cut2_4xmax) ) and ( cX < xmax ) and
             ( cY > ymin + int(cut1_4xmax) ) and ( cY < ymin + int(cut2_4ymax) ) ): on[2] = 1
        if ( ( cX > xmin + int(cut1_4xmax) ) and ( cX < xmin + int(cut3_4xmax) ) and
             ( cY > ymin + int(cut1_4ymax) ) and ( cY < ymin + int(cut3_4ymax) ) ): on[3] = 1
        if ( ( cX > xmin ) and ( cX < xmin + int(cut1_4xmax) ) and
             ( cY > ymin + int(cut2_4ymax) ) and ( cY < ymax ) ): on[4] = 1
        if ( ( cX > xmax - int(cut1_4xmax) ) and ( cX < xmax ) and
             ( cY > ymin + int(cut2_4ymax) ) and ( cY < ymax) ): on[5] = 1
        if ( ( cX > xmin ) and ( cX < xmax ) and
             ( cY > ymax - int(cut1_4ymax) ) and ( cY < ymax ) ): on[6] = 1
    
    number = DIGITS_LOOKUP.get(tuple(on))
    if ( number == None ):
        number = -1;
    #number = DIGITS_LOOKUP[tuple(on)]
    return number

def removeCircles( counters ):
    cnts = []
    xmax = 0
    ymax = 0
    for c in counters:
        (x, y, w, h) = cv2.boundingRect(c) # Compute the bounding box of the contour
        xmaxAux = w
        ymaxAux = h
        if ( xmax < xmaxAux ): xmax = xmaxAux
        if ( ymax < ymaxAux ): ymax = ymaxAux
    for c in counters:
        (x, y, w, h) = cv2.boundingRect(c) # Compute the bounding box of the contour
        if ( w > xmax/2 ) or ( h > ymax/2 ):
            cnts.append( c )
    return cnts, xmax, ymax

def detection( file ):
    cnts = cv2.findContours(file, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if ( len( cnts ) == 2 ): # version opencv 4.x
        cnts = cnts[0]
    else: # version opencv 3, 4-pre, or 4-alpha
        cnts = cnts[1]
    cnts = sort_contours(cnts, method="left-to-right")[0]

    digit = []
    numbers = []

    cnts, xmaxAux, ymaxAux = removeCircles( cnts )
    
    # analisa se os filamentos formam um digito
    if ( len(cnts) > 0 ): # there is contours
        counter = 0 # counter contours
        while (counter < len( cnts )):
            idNumber = False; contInt = counter
            while (( idNumber == False ) or (contInt < len(cnts))):
                digit.append( cnts[contInt] )
                (xmin, ymin, xmax, ymax) = delimitRegion( file, digit )
                #print xmin, ymin, xmax, ymax
                #cv2.rectangle(file, (xmin, ymin), (xmax, ymax), (255, 255, 255), 1)
                #cv2.imshow( "detected", file )
                #cv2.waitKey( 0 )
                # Condicao para encontrar o 1
                if (( len(digit) == 2 ) and ( contInt+1 < len(cnts) ) ):
                    (x, y, w, h) = cv2.boundingRect(cnts[contInt+1]) # Compute the bounding box of the contour
                    if ( x > xmax + xmaxAux*0.10 ): # se a posicao do proximo filamento esta nao esta a 10% do tamanho do filamento 
                        number = 1;
                else:
                    number = ( identifyNumber( file, digit, xmin, ymin, xmax, ymax ) )

                if ( contInt+1 < len(cnts) ):
                    (x, y, w, h) = cv2.boundingRect(cnts[contInt+1]) # Compute the bounding box of the contour
                    if ( x < xmax + xmaxAux*0.10): # se a posicao do proximo filamento esta nao esta a 10% do tamanho do filamento 
                        number = -1

                if ( number <> -1 ):
                    #print number
                    numbers.append( number )
                    digit = []
                    xmin = 0; ymin = 0; xmax = 0; ymax = 0
                    idNumber = True
                contInt += 1
            counter += contInt if contInt > 0 else 1
            contInt = 0
        
    return numbers


if __name__ == '__main__':
    file = sys.argv[1]
    img = cv2.imread( file )

    edged = filterImage( img )
    cv2.imshow( "edged", edged )
    cv2.waitKey( 0 )
    
    numbers = detection( edged )
    for i in numbers:
        print i
    
