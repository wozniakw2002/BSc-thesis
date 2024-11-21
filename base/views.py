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
import numpy as np

def mainPage(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                uploaded_file = request.FILES['image']
                image = Image.open(uploaded_file)

                # Przetwarzanie obrazu
                processed_image = DataPreparation.single_photo_preparation(image=image)
                
                # Pobranie modelu i wykonanie predykcji
                model = ModelSingleton.get_model()
                prediction, label = Model.predict(model, processed_image)

                # Funkcja pomocnicza do konwersji obrazu na base64
                def convert_to_base64(image_array):
                    final_image = Image.fromarray(
                        np.array((image_array * 255).tolist(), dtype=np.uint8).squeeze()
                    )
                    buffered = BytesIO()
                    final_image.save(buffered, format="PNG")
                    return base64.b64encode(buffered.getvalue()).decode('utf-8')

                # Przygotowanie kontekstu w zależności od liczby obrazów
                if len(processed_image) == 1:
                    img_str = convert_to_base64(processed_image)
                    context = {
                        'image_1': img_str,
                        'prediction_1': prediction[0][0],
                        'label_1': label[0][0]
                    }
                else:
                    img_str_1 = convert_to_base64(processed_image[0])
                    img_str_2 = convert_to_base64(processed_image[1])
                    context = {
                        'image_1': img_str_1,
                        'image_2': img_str_2,
                        'prediction_1': prediction[0][0],
                        'prediction_2': prediction[1][0],
                        'label_1': label[0][0],
                        'label_2': label[1][0]
                    }

                return render(request, 'result.html', context)

            except Exception as e:
                # Obsługa błędów podczas przetwarzania obrazu
                return render(request, 'mainPage.html', {
                    'form': form,
                    'error': f"An error occurred while processing the image: {str(e)}"
                })

    else:
        form = ImageUploadForm()

    return render(request, 'mainPage.html', {'form': form})

def generate_pdf(request):
    # Pobierz dane z requesta
    prediction_1 = request.GET.get('prediction_1')
    label_1 = request.GET.get('label_1')
    image_1_data = request.GET.get('image_1')

    prediction_2 = request.GET.get('prediction_2')
    label_2 = request.GET.get('label_2')
    image_2_data = request.GET.get('image_2')

    # Funkcja pomocnicza do zapisywania obrazów z base64
    def save_temp_image(image_data, file_name):
        decoded_image = base64.b64decode(image_data)
        temp_path = os.path.join(settings.MEDIA_ROOT, file_name)
        with open(temp_path, 'wb') as temp_img_file:
            temp_img_file.write(decoded_image)
        return temp_path

    # Zapisz obrazy tymczasowe
    temp_image_1_path = save_temp_image(image_1_data, 'temp_image_1.png')
    temp_image_2_path = None
    if image_2_data:
        temp_image_2_path = save_temp_image(image_2_data, 'temp_image_2.png')

    # Przygotowanie odpowiedzi jako plik PDF
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="prediction_report.pdf"'

    # Utwórz dokument PDF
    p = canvas.Canvas(response, pagesize=letter)
    width, height = letter

    # Dodaj nagłówek
    p.setFont("Helvetica-Bold", 16)
    p.drawString(100, height - 50, "Prediction Report")

    # Wyświetl pierwszy obraz z predykcją i etykietą
    y_position = height - 100
    p.setFont("Helvetica", 12)
    p.drawString(100, y_position, f"Prediction 1: {prediction_1}")
    y_position -= 20
    p.drawString(100, y_position, f"Label 1: {label_1}")
    y_position -= 220  # Przesunięcie na obraz
    p.drawImage(temp_image_1_path, 100, y_position, width=4*inch, height=3*inch)

    # Wyświetl drugi obraz z predykcją i etykietą (jeśli istnieje)
    if temp_image_2_path:  # Przesunięcie na kolejną sekcję
        y_position -= 240
        p.drawString(100, y_position + 200, f"Prediction 2: {prediction_2}")
        p.drawString(100, y_position + 180, f"Label 2: {label_2}")
        y_position -= 40
        p.drawImage(temp_image_2_path, 100, y_position, width=4*inch, height=3*inch)

    # Zakończ stronę PDF
    p.showPage()
    p.save()

    # Usuń tymczasowe obrazy
    os.remove(temp_image_1_path)
    if temp_image_2_path:
        os.remove(temp_image_2_path)

    return response


def modelInfo(request):
    return render(request, 'modelInfo.html')

def userGuide(request):
    return render(request, 'userGuide.html')
