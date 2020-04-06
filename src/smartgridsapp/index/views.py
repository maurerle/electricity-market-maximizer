from django.shortcuts import render
from django.contrib.auth import logout
# Create your views here.
def index(request):
    if request.user.is_authenticated:
        logout(request)
    return render(request, "index/index.html")