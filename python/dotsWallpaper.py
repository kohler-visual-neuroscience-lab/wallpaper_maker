# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 10:55:52 2020

@author: Chris
"""

import numpy as np
import cairo as cr
import math


def genDotsFund(size, minRad, maxRad, numOfDots, wptype):
    #radius = 0.1;
    #size = 500;
    height = 0;
    width = 0;
    #get correct size of dots tile based on wallpaper chosen
    if (wptype == "P1"):
        height = size;
        width = size;
    elif (wptype == "P2" or wptype == "PM" or wptype == "PG"):
        width = round(size / 2);
        height = 2 * width;
    elif (wptype == "PMM" or wptype == "CM" or wptype == "PMG" or wptype == 'PGG' or wptype == 'P4' or wptype == 'P4M' or wptype == 'P4G'):
        height = round(size / 2);
        width = height;
    elif (wptype == 'P3'):
        alpha = np.pi/3;
        s = size  / math.sqrt(3 * np.tan(alpha));
        height = math.floor(s * 1.5) * 6;
        width = size * 6;
    elif (wptype == 'P3M1'):
        alpha = np.pi/3;
        s = size / math.sqrt(3 * np.tan(alpha));
        height = round(s) * 10;
        width = size * 10;
    elif (wptype == 'P31M' or wptype == 'P6'):
        s = size / math.sqrt(math.sqrt(3));
        height = round(s) * 6;
        width = size * 6;
    elif (wptype == 'P6M'):
        s = size / math.sqrt(math.sqrt(3));
        height = round(s/2) * 10;
        width = size * 10;
    elif (wptype == 'CMM'):
        width = round(size/4);
        height = 2*width;
        
    surface = cr.ImageSurface(cr.FORMAT_ARGB32, width, height);
    ctx = cr.Context(surface);
    ctx.scale(width, height);

    pat = cr.SolidPattern(0.5, 0.5, 0.5, 1.0)



    ctx.rectangle(0, 0, width, height)  # Rectangle(x0, y0, x1, y1)
    ctx.set_source(pat)
    ctx.fill()



    #generate random dots
    #numOfDots = 10;
    previousDots = [];
    for x in range(numOfDots):
        isPossibleDot = False;
        while isPossibleDot == False:
            radius = np.random.uniform(minRad, maxRad);
            xc = np.random.uniform(radius, 1-radius);   
            yc = np.random.uniform(radius, 1-radius);
            if (wptype == 'P3'):
                xc = np.random.uniform(radius, 1-max(0.75, radius * 2));   
                yc = np.random.uniform(0.30 + radius, 1-max(0.35, radius * 2));
            elif (wptype == 'P3M1'):
                xc = np.random.uniform(0.025 + radius, 1-max(0.93, radius * 2));   
                yc = np.random.uniform(0.30 + radius, 1-max(0.45, radius * 2));
            elif (wptype == 'P31M'):
                xc = np.random.uniform(0.025 + radius, 1-max(0.93, radius * 2));   
                yc = np.random.uniform(0.30 + radius, 1-max(0.45, radius * 2));
            elif (wptype == 'P6'):
                xc = np.random.uniform(0.025 + radius, 1-max(0.93, radius * 2));   
                yc = np.random.uniform(0.30 + radius, 1-max(0.45, radius * 2));
            elif (wptype == 'P6M'):
                xc = np.random.uniform(0.025 + radius, 1-max(0.93, radius * 2));   
                yc = np.random.uniform(0.30 + radius, 1-max(0.45, radius * 2));

            
            if x == 0:
                ctx.set_source_rgb(0, 0, 0)  # Solid color
                ctx.arc(xc, yc, radius, 0, 2*math.pi) #circle
                ctx.fill();
                previousDots.append([xc,yc, radius]);
                isPossibleDot = True;
            else:
                for y in previousDots:
                    d = (xc - y[0])**2 + (yc-y[1])**2;
                    radSumSq = (radius + y[2])**2;
                    if (d > radSumSq):
                        isPossibleDot = True;
                    else:
                        isPossibleDot = False;
                        break;
            if isPossibleDot == True:
                ctx.set_source_rgb(0 + radius * 10, 0 + radius * 10, 0 + radius * 10)  # Solid color
                ctx.arc(xc, yc, radius, 0, 2*math.pi) #circle0
                ctx.fill();
                previousDots.append([xc,yc, radius]);
        

    # Arc(cx, cy, radius, start_angle, stop_angle)
    #ctx.arc(0.2, 0.1, 0.1, -math.pi / 2, 0)
    #ctx.line_to(0.5, 0.1)  # Line to (x,y)
    # Curve(x1, y1, x2, y2, x3, y3)
    #ctx.curve_to(0.5, 0.2, 0.5, 0.4, 0.2, 0.8)
    #ctx.close_path()


    #ctx.set_line_width(0.02)

    #ctx.stroke()


    surface.write_to_png("example.png");
    buf = surface.get_data()
    if (wptype == 'P3' or wptype == 'P3M1' or wptype == 'P31M' or wptype == 'P6' or wptype == 'P6M'):
        result = np.ndarray(shape=(height, width),dtype=np.uint32,buffer=buf)
    else:
        result = np.ndarray(shape=(width, height),dtype=np.uint32,buffer=buf)
    return result;
