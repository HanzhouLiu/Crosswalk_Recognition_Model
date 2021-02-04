"""
This function is used to calculate hausdorff distance
A&B locate on the same line
C&D locate on the same line
"""
import numpy as np
import math_tools as mt
import math


def hd(A, B, C, D):
    vector_AB = B - A
    vector_CD = D - C

    hd1 = 2*mt.tri_area(A, B, C)/math.sqrt(vector_AB[0]**2+vector_AB[1]**2)
    hd2 = 2*mt.tri_area(A, B, D)/math.sqrt(vector_AB[0]**2+vector_AB[1]**2)
    hd3 = 2*mt.tri_area(A, C, D)/math.sqrt(vector_CD[0]**2+vector_CD[1]**2)
    hd4 = 2*mt.tri_area(B, C, D)/math.sqrt(vector_CD[0]**2+vector_CD[1]**2)
    hd_avr1 = (hd1 + hd2)/2
    hd_avr2 = (hd3 + hd4)/2
    return hd_avr1, hd_avr2
