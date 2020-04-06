from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect
from register.models import Profile
from .backend import influxClient as influx
from .backend import genetic as ga
from .forms import Production
from threading import Thread
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.http import StreamingHttpResponse
import json

LIMIT = {}

@login_required(login_url='/register/')
def mgp(request):
    data = Profile.objects.filter(user = request.user)
    
    for d in data:
        dp, dq, op, oq = influx.getData('MGP', d.operator)
    
    context = {'data':data, 'dp':dp, 'dq':dq, 'op':op, 'oq':oq}
        
    return render(request, 'dashboard/mgp.html', context)


@login_required(login_url='/register/')
def mi(request):
    data = Profile.objects.filter(user = request.user)
    
    for d in data:
        dp, dq, op, oq = influx.getData('MI', d.operator)
    
    context = {'data':data, 'dp':dp, 'dq':dq, 'op':op, 'oq':oq}
    
    return render(request, 'dashboard/mi.html', context)

@login_required(login_url='/register/')
def msd(request):
    data = Profile.objects.filter(user = request.user)
    
    for d in data:
        dp, dq, op, oq = influx.getData('MSD', d.operator)
    
    context = {'data':data, 'dp':dp, 'dq':dq, 'op':op, 'oq':oq}
    
    return render(request, 'dashboard/msd.html', context)


@login_required(login_url='/register/')
def optimize(request):
    print(request.method)
    if request.method == 'GET':
        
        data = Profile.objects.filter(user = request.user)
        form = Production()
        context = {'data':data, 'form':form}    
        
        return render(request, 'dashboard/optimize.html', context)

    elif request.method == 'POST':
        data = Profile.objects.filter(user = request.user)
        for d in data:
            LIMIT[d.operator] = float(request.POST['prod_limit'])
            print(LIMIT)
        return render(request, 'dashboard/wait.html')


@login_required(login_url='/register/')
def wait(request):
    print(request.GET)
    return render(request, 'dashboard/wait.html')


@login_required(login_url='/register/')
def process(request):
    data = Profile.objects.filter(user = request.user)
    
    for d in data:
        genetic = ga.Genetic(d.operator, LIMIT[d.operator])
        profit, sol = genetic.run()
        profit = profit
        MGPdp = round(sol[0][0][2], 2)
        MGPdq = round(sol[0][0][3], 2)
        MGPop = round(sol[0][0][0], 2)
        MGPoq = round(sol[0][0][1], 2)
        MIdp = round(sol[0][0][6], 2)
        MIdq = round(sol[0][0][7], 2)
        MIop = round(sol[0][0][4], 2)
        MIoq = round(sol[0][0][5], 2)
        MSDdp = round(sol[0][0][10], 2)
        MSDdq = round(sol[0][0][11], 2)
        MSDop = round(sol[0][0][8], 2)
        MSDoq= round(sol[0][0][9], 2)
    
    context = {
        'data':data, 
        'profit':profit, 
        'MGPdp':MGPdp, 
        'MGPdq':MGPdq, 
        'MGPop':MGPop, 
        'MGPoq':MGPoq, 
        'MIdp':MIdp, 
        'MIdq':MIdq, 
        'MIop':MIop, 
        'MIoq':MIoq, 
        'MIdp':MSDdp, 
        'MSDdq':MSDdq, 
        'MSDop':MSDop, 
        'MSDoq':MSDoq,
        'MSDdp':MSDdp
    }
    print(context)
    return render(request, 'dashboard/optimized.html', context)