"""smartgridsapp URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from register import views as vr
from index import views as vi
from dashboard import views as vd

urlpatterns = [
    path('admin/', admin.site.urls),
    path('register/', vr.register, name='register'),
    path('', vi.index, name='index'),
    path('dashboard/mgp', vd.mgp, name='mgp'),
    path('dashboard/mi', vd.mi, name='mi'),
    path('dashboard/msd', vd.msd, name='mi'),
    path('dashboard/optimize', vd.optimize, name='opt'),
    path('dashboard/wait', vd.wait, name='wait'),
    path('dashboard/results', vd.process, name='process')
    path('login/', vr.loginPage, name='login'),
    path('logout/', vr.logoutUser, name='logout'),
]