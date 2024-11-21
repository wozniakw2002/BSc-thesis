from django.urls import path
from . import views

urlpatterns = [
    path('', views.mainPage, name="main"),
    path('modelInfo/', views.modelInfo, name="modelInfo"),
    path('userGuide/', views.userGuide, name="userGuide"),
   path('generate_pdf/', views.generate_pdf, name='generate_pdf'),
]