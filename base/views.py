from django.shortcuts import render
from django.http import JsonResponse
from Modules.ModelModule import DataPreparation

def mainPage(request):
    return render(request, 'mainPage.html')

def modelInfo(request):
    return render(request, 'modelInfo.html')
