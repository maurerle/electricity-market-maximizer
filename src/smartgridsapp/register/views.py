from django.shortcuts import render
from .forms import CreateUserForm
from django.contrib.auth.models import User
from .models import Profile
from django.contrib.auth import authenticate, login, logout
from django.shortcuts import redirect
from django.urls import reverse
# Create your views here.
def register(request, reason=''):
    if request.user.is_authenticated:
        logout(request)

    if request.method == 'POST':
        form = CreateUserForm(request.POST)
            
        if form.is_valid():
            username = request.POST['username']
            password = request.POST['password1']
            operator = request.POST['operator']
            
            user = User.objects.create_user(
                username = username,
                password = password,
            )

            new_user = Profile(operator=operator, user=user)
            new_user.save()

            user = authenticate(username=username, password=password)
            if user:
                if user.is_active:
                    login(request, user)
                    return redirect('../dashboard/mgp')
    
    
    elif request.method == 'GET':
        form = CreateUserForm()
    context = {'form':form}
    
    
    return render(request, "register/register.html", context)

"""
def csrf_failure(request, reason=""):
    if request.user.is_authenticated:
        logout(request)
        register(request)
    #context = {'form': CreateUserForm(request.POST)}
    #return render(request, 'register/register.html', context)
"""