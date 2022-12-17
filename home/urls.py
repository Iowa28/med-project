from django.urls import path
from django.contrib.auth import views

from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('news.html', views.news,name ='news'),
    path('login', views.loginPage,name='login'),
    path('register', views.registerPage, name='register'),
    path('logout', views.logoutUser, name='logout'),
    path('personal-cab-3.html', views.personalcab,name='personalcab'),
    path('personal-cab-3-changedata.html', views.personalcab_changedata, name='changedata'),
    path('analyse.html', views.analyse,name='analyse'),
    path('result.html', views.getresults, name='result'),
    path('anemia-analyse', views.anemia_analyse, name='anemiaanalyse'),
    path('anemia-result', views.anemia_result, name='anemiaresult')
]
