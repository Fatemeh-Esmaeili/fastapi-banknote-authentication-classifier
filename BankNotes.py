# -*- coding: utf-8 -*-
"""
Created on Tue Aug 08 2025 16:45
"""
from pydantic import BaseModel
# 2. Class which describes Bank Notes measurements
class BankNote(BaseModel):
    variance: float
    skewness: float
    curtosis: float
    entropy: float
