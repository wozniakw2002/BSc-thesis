from django.shortcuts import render
from PIL import Image
from Modules.ModelModule import DataPreparation, Model 
from Modules.GradCam import GradCam 
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
from reportlab.lib import colors

def mainPage(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
          
            uploaded_file = request.FILES['image']
            image = Image.open(uploaded_file)

        
            processed_image = DataPreparation.single_photo_preparation(image=image)
            
            model = ModelSingleton.get_model()
            prediction, label = Model.predict(model, processed_image)

            
            def convert_to_base64(image):
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                return base64.b64encode(buffered.getvalue()).decode('utf-8')
            

            
            if len(processed_image) == 1:
                img_str = convert_to_base64(GradCam.create_and_overlap_gradcam(processed_image, processed_image[0],model))
                context = {
                    'prediction_1': prediction[0][0],
                    'label_1': label[0][0],
                    'image_1': img_str
                }
            else:
                img_str_1 = convert_to_base64(GradCam.create_and_overlap_gradcam(np.array([processed_image[0]]), processed_image[0],model))
                img_str_2 = convert_to_base64(GradCam.create_and_overlap_gradcam(np.array([processed_image[1]]), processed_image[1],model))

                context = {
                    'prediction_1': prediction[0][0],
                    'prediction_2': prediction[1][0],
                    'label_1': label[0][0],
                    'label_2': label[1][0],
                    'image_1': img_str_1,
                    'image_2': img_str_2
                }

            return render(request, 'result.html', context)

        

    else:
        form = ImageUploadForm()

    return render(request, 'mainPage.html', {'form': form})

def generate_pdf(request):
    if request.method == 'POST':
        prediction_1 = request.POST.get('prediction_1')
        label_1 = request.POST.get('label_1')
        image_1_data = request.POST.get('image_1')

        prediction_2 = request.POST.get('prediction_2')
        label_2 = request.POST.get('label_2')
        image_2_data = request.POST.get('image_2')

        def save_temp_image(image_data, file_name):
            decoded_image = base64.b64decode(image_data)
            temp_path = os.path.join(settings.MEDIA_ROOT, file_name)
            with open(temp_path, 'wb') as temp_img_file:
                temp_img_file.write(decoded_image)
            return temp_path

        temp_image_1_path = save_temp_image(image_1_data, 'temp_image_1.png')
        temp_image_2_path = None
        if image_2_data:
            temp_image_2_path = save_temp_image(image_2_data, 'temp_image_2.png')

        response = HttpResponse(content_type='application/pdf')
        response['Content-Disposition'] = 'attachment; filename="prediction_report.pdf"'

        p = canvas.Canvas(response, pagesize=letter)
        width, height = letter

        p.setFont("Helvetica-Bold", 16)
        p.drawString(100, height - 50, "Prediction Report")

        y_position = height - 100
        p.setFont("Helvetica", 12)

        if label_1 == "1":
            p.setFillColor(colors.red)
            if temp_image_2_path:
                p.drawString(100, y_position, f"Left knee is unhealthy - Probability of disease: {float(prediction_1):.2f}")
            else:
                p.drawString(100, y_position, f"Knee is unhealthy - Probability of disease: {float(prediction_1):.2f}")
        else:
            p.setFillColor(colors.green)
            if temp_image_2_path:
                p.drawString(100, y_position, f"Left knee is healthy - Probability of disease: {float(prediction_1):.2f}")
            else:
                p.drawString(100, y_position, f"Knee is healthy - Probability of disease: {float(prediction_1):.2f}")

        y_position -= 20
        p.setFillColor(colors.black)  
        y_position -= 220 
        p.drawImage(temp_image_1_path, 100, y_position, width=4*inch, height=3*inch)


        if temp_image_2_path:  
            y_position -= 240
            if label_2 == "1":
                p.setFillColor(colors.red)
                p.drawString(100, y_position + 200, f"Right knee is unhealthy - Probability of disease: {float(prediction_2):.2f}")
            else:
                p.setFillColor(colors.green)
                p.drawString(100, y_position + 200, f"Right knee is healthy - Probability of disease: {float(prediction_2):.2f}")
            p.setFillColor(colors.black)  
            y_position -= 40
            p.drawImage(temp_image_2_path, 100, y_position, width=4*inch, height=3*inch)

    
        p.showPage()
        p.save()

        
        os.remove(temp_image_1_path)
        if temp_image_2_path:
            os.remove(temp_image_2_path)

        return response



def modelInfo(request):
    return render(request, 'modelInfo.html')

def userGuide(request):
    return render(request, 'userGuide.html')
