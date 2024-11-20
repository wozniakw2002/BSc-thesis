from django.shortcuts import render
from PIL import Image
from Modules.ModelModule import DataPreparation, Model  
from .forms import ImageUploadForm
from .singleton import ModelSingleton
import base64
from io import BytesIO
from django.http import HttpResponse
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os
from django.conf import settings
from reportlab.lib.units import inch

def mainPage(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES['image']
            image = Image.open(uploaded_file)
        
            # Przetwarzanie obrazu
            processed_image = DataPreparation.single_photo_preparation(image=image)
            
            # Predykcja za pomocą modelu
            model = ModelSingleton.get_model()
            prediction, label = Model.predict(model, processed_image)
            
            # Przechowanie obrazu w pamięci jako base64 (do późniejszego wyświetlenia)
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # Przesyłamy dane do kontekstu, aby pokazać je na stronie
            context = {
                'image': img_str, 
                'prediction': prediction[0][0],
                'label': label[0][0]
            }
            return render(request, 'result.html', context)
    else:
        form = ImageUploadForm()

    return render(request, 'mainPage.html', {'form': form})

def generate_pdf(request):
    prediction = request.GET.get('prediction')
    label = request.GET.get('label')
    image_data = request.GET.get('image_data')

    image_data = base64.b64decode(image_data)

    # Tworzymy tymczasowy plik obrazu w MEDIA_ROOT
    temp_image_path = os.path.join(settings.MEDIA_ROOT, 'temp_image.png')
    
    # Zapisujemy obrazek tymczasowo
    with open(temp_image_path, 'wb') as temp_img_file:
        temp_img_file.write(image_data)

    
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="prediction_report.pdf"'

    
    p = canvas.Canvas(response, pagesize=letter)
    width, height = letter  # wymiary strony

    # Dodajemy tytuł
    p.setFont("Helvetica-Bold", 16)
    p.drawString(100, height - 100, "Prediction Report")

    # Dodajemy pierwszy napis
    p.setFont("Helvetica", 12)
    p.drawString(100, height - 130, f"Label: {prediction}")

    # Dodajemy drugi napis
    p.drawString(100, height - 160, f"Label: {label}")

    # Dodajemy obrazek

    p.drawImage(temp_image_path, 100, height - 400, width=4*inch, height=3*inch)

    # Zakończ rysowanie PDF
    p.showPage()
    p.save()

    os.remove(temp_image_path)

    return response


def modelInfo(request):
    return render(request, 'modelInfo.html')

def userGuide(request):
    return render(request, 'userGuide.html')
