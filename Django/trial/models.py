from django.db import models
from django.forms import ModelForm
from math import pi

class Input(models.Model):
    demand = models.CharField(
        verbose_name=' Demand',default='3,3,2,2,1,1,3,3',max_length=100)
    isl = models.CharField(
        verbose_name=' Initial Stock Level',default='8,21,4,3,15,4,35,3',max_length=100)
    weekp = models.TextField(
        verbose_name=' Weekly_penalty',default='1,1,10,2,4,2,2,1,2,2,3,1,2,2,3,1,4,4,3,1,2,3,4,1,4,4,3,1,4,7,3,1')
    tr = models.IntegerField(
        verbose_name=' Total number of rakes to be allocated')
    iterations = models.IntegerField(
        verbose_name=' Maximun number of iterations')     
    combm = models.TextField(
        verbose_name=' Combination Matrix',default='0,1,0,0,0,1,0,1,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,1,0,1,1,0,0,0,0,1,1,0')
    n = models.IntegerField(
        verbose_name=' Number Of Weeks for which allocation has to be made',default=4)
    m = models.IntegerField(
        verbose_name=' Number Of Warehouses for which allocation has to be made',default=8)
    sc = models.CharField(
        verbose_name=' Storage Capacity of each Warehouse',default='13,25,6,5,16,5,40,6',max_length=100)
    st = models.CharField(
        verbose_name=' Storage Capacity of each Terminal',default='2,2,2,2,2,2,2,2',max_length=100)
    rph = models.IntegerField(
        verbose_name=' Half rake penalty',default=20)
    rpf = models.IntegerField(
        verbose_name=' Full rake penalty',default=50)
    ma = models.CharField(
        verbose_name=' Maximum Number of Rakes that can be allocated to each warehouse',default='8,8,8,8,8,8,8,8',max_length=100)
    kshift = models.IntegerField(
        verbose_name=' K_shift',default=135)
    kterm = models.IntegerField(verbose_name=' K_terminate',default=7450)
    kreset = models.IntegerField(verbose_name=' K_reset',default=575)

class InputForm(ModelForm):
    class Meta:
        model = Input
        fields = "__all__" 
       