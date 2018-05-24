from django.shortcuts import render,render_to_response
from django.template import RequestContext
from django.http import HttpResponse
from .forms import FirstForm,SecForm
from trial.compute import compute
import os
def index(request):
   
    os.chdir(os.path.dirname(__file__))
    
    form = FirstForm(prefix='main')
    sub_form = SecForm(prefix='sub')
    
    if request.method == 'POST':
        form = FirstForm(request.POST, prefix="main")
        sub_form = SecForm(request.POST, prefix="sub")
        if form.is_valid():
            demand = form.cleaned_data['demand']
            isl = form.cleaned_data['isl']
            tr = form.cleaned_data['tr']
            iterations = form.cleaned_data['iterations']
            weekp = form.cleaned_data['weekp'] 
        if sub_form.is_valid():    
            combm = sub_form.cleaned_data['combm']
            n = sub_form.cleaned_data['n']
            m = sub_form.cleaned_data['m']
            sc = sub_form.cleaned_data['sc']
            st = sub_form.cleaned_data['st']
            rph = sub_form.cleaned_data['rph']
            rpf = sub_form.cleaned_data['rpf']
            ma = sub_form.cleaned_data['ma']
            kshift = sub_form.cleaned_data['kshift']
            kreset = sub_form.cleaned_data['kreset']
            kterm = sub_form.cleaned_data['kterm']
            result = compute(demand,isl, weekp, tr, iterations, combm, n, m, sc, st, rph, rpf, ma, kshift, kterm, kreset)       
            
    else:
        form = FirstForm(prefix="main")
        sub_form = SecForm(prefix="sub")

    return render(request,'trial.html',
            {'form': form,
             'sub_form':sub_form,
             },)
def display(request):
    result=static()
    return render(request,'trial1.html',{'result':result,},)