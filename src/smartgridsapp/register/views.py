from django.shortcuts import render
from .forms import CreateUserForm
from django.contrib.auth.models import User
from .models import Profile
# Create your views here.
def register(request):
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
    
    
    elif request.method == 'GET':
        form = CreateUserForm()
    context = {'form':form}
    
    
    return render(request, "register/register.html", context)

