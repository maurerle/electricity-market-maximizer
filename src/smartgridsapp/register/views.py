from django.shortcuts import render, redirect
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
            messages.success(request, f'Account was created for {username}')

            return redirect('login')
    
    
    elif request.method == 'GET':
        form = CreateUserForm()
    context = {'form':form}
    
    
    return render(request, "register/register.html", context)
   

def loginPage(request):

    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        print (username)
        print (password)


        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect('register')
        else:
            messages.info(request, 'Username OR Password is not correct')
            
    
            
    context={}
    return render(request, 'register/login.html', context)

def logoutUser(request):
    logout(request)
    return redirect('login')