from django.shortcuts import render
from PIL import Image
from Modules.ModelModule import DataPreparation, Model  
from .forms import ImageUploadForm
from .singleton import ModelSingleton

def mainPage(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():

            uploaded_file = request.FILES['image']

            image = Image.open(uploaded_file)

            processed_image = DataPreparation.single_photo_preparation(image=image)

            model = ModelSingleton.get_model()
            prediction, label = Model.predict(model,processed_image)

            return render(request, 'result.html', {'prediction': prediction[0], 'label': label[0]})
    else:
        form = ImageUploadForm()
    return render(request, 'mainPage.html', {'form': form})


def modelInfo(request):
    return render(request, 'modelInfo.html')
