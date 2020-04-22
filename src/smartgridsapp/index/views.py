from django.shortcuts import render
from django.contrib.auth import logout
# Create your views here.
def index(request):
<<<<<<< HEAD
    if request.user.is_authenticated:
        logout(request)
    return render(request, "index/index.html")
=======
    
    return render(request, "index/index.html")

def about(request):

    return render(request, "index/dataSource.html")
>>>>>>> Contacts
