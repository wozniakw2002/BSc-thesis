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
from Modules.InterpretationModule import Interpretation

def mainPage(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
          
            uploaded_file = request.FILES['image']
            image = Image.open(uploaded_file)

        
            processed_image = DataPreparation.single_photo_preparation(image=image)
            
            model = ModelSingleton.get_model()
            prediction, label = Model.predict(model, processed_image)

            
            visualizations = request.POST.getlist('visualizations')

            def convert_to_base64(image):
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                return base64.b64encode(buffered.getvalue()).decode('utf-8')

            context = {}

   
            
            for i in range(len(processed_image)):
    
                if 'gradcam' in visualizations:
                    GradCam_str = convert_to_base64(GradCam.create_and_overlap_gradcam(np.array([processed_image[i]]), processed_image[i], model))
                    context[f'GradCam_image_{i+1}'] = GradCam_str

                
                if 'lime' in visualizations:
                    Lime_str = convert_to_base64(Interpretation.show_lime_interpretation(model, np.array([processed_image[i]]), 1))
                    context[f'Lime_image_{i+1}'] = Lime_str

            
                if 'shap' in visualizations:
                    Shap_str = convert_to_base64(Interpretation.show_shap(model, np.array([processed_image[i]])))
                    context[f'ShapValues_image_{i+1}'] = Shap_str

        
                context[f'prediction_{i+1}'] = prediction[i][0]
                context[f'label_{i+1}'] = label[i][0]

            return render(request, 'result.html', context)

    else:
        form = ImageUploadForm()

    return render(request, 'mainPage.html', {'form': form})

def generate_pdf(request):
    if request.method == 'POST':
        prediction_1 = request.POST.get('prediction_1')
        label_1 = request.POST.get('label_1')
        
        grad_image_1_data = request.POST.get('GradCam_image_1')
        lime_image_1_data = request.POST.get('Lime_image_1')
        shap_image_1_data = request.POST.get('ShapValues_image_1')

        prediction_2 = request.POST.get('prediction_2')
        label_2 = request.POST.get('label_2')
        
        grad_image_2_data = request.POST.get('GradCam_image_2')
        lime_image_2_data = request.POST.get('Lime_image_2')
        shap_image_2_data = request.POST.get('ShapValues_image_2')

        def save_temp_image(image_data, file_name):
            if not image_data:
                return None
            decoded_image = base64.b64decode(image_data)
            temp_path = os.path.join(settings.MEDIA_ROOT, file_name)
            with open(temp_path, 'wb') as temp_img_file:
                temp_img_file.write(decoded_image)
            return temp_path

        temp_grad_1_path = save_temp_image(grad_image_1_data, 'grad_image_1.png') if grad_image_1_data else None
        temp_lime_1_path = save_temp_image(lime_image_1_data, 'lime_image_1.png') if lime_image_1_data else None
        temp_shap_1_path = save_temp_image(shap_image_1_data, 'shap_image_1.png') if shap_image_1_data else None

        temp_grad_2_path = save_temp_image(grad_image_2_data, 'grad_image_2.png') if grad_image_2_data else None
        temp_lime_2_path = save_temp_image(lime_image_2_data, 'lime_image_2.png') if lime_image_2_data else None
        temp_shap_2_path = save_temp_image(shap_image_2_data, 'shap_image_2.png') if shap_image_2_data else None

        response = HttpResponse(content_type='application/pdf')
        response['Content-Disposition'] = 'attachment; filename="prediction_report.pdf"'

        p = canvas.Canvas(response, pagesize=letter)
        width, height = letter

        p.setFont("Helvetica-Bold", 16)
        p.drawString(100, height - 50, "Prediction Report")

        y_position = height - 100
        p.setFont("Helvetica-Bold", 14)

        if label_1 == "1":
            p.setFillColor(colors.red)
            p.drawString(100, y_position, f"Left knee is unhealthy - Probability of disease: {float(prediction_1):.2f}")
        else:
            p.setFillColor(colors.green)
            p.drawString(100, y_position, f"Left knee is healthy - Probability of disease: {float(prediction_1):.2f}")

        def check_and_add_new_page(h=250):
            nonlocal y_position
            if y_position < h:
                p.showPage()
                y_position = height

        if temp_grad_1_path:
            check_and_add_new_page()
            y_position -= 240
            p.drawImage(temp_grad_1_path, 100, y_position)

        if temp_lime_1_path:
            check_and_add_new_page()
            y_position -= 240
            p.drawImage(temp_lime_1_path, 100, y_position)
            
        if temp_shap_1_path:
            check_and_add_new_page(390)
            y_position -= 380
            p.drawImage(temp_shap_1_path, 80, y_position)

        if prediction_2:
            check_and_add_new_page(300)
            p.setFont("Helvetica-Bold", 14)
            y_position -= 50
            if label_2 == "1":
                p.setFillColor(colors.red)
                p.drawString(100, y_position, f"Right knee is unhealthy - Probability of disease: {float(prediction_2):.2f}")
            else:
                p.setFillColor(colors.green)
                p.drawString(100, y_position, f"Right knee is healthy - Probability of disease: {float(prediction_2):.2f}")
            
            
            if temp_grad_2_path:
                y_position -= 240
                p.drawImage(temp_grad_2_path, 100, y_position)
                
            if temp_lime_2_path:
                check_and_add_new_page()
                y_position -= 240
                p.drawImage(temp_lime_2_path, 100, y_position)
            
            if temp_shap_2_path:
                check_and_add_new_page(390)
                y_position -= 380
                p.drawImage(temp_shap_2_path, 80, y_position)

        p.showPage()
        p.save()

        if temp_grad_1_path:
            os.remove(temp_grad_1_path)
        if temp_lime_1_path:
            os.remove(temp_lime_1_path)
        if temp_shap_1_path:
            os.remove(temp_shap_1_path)
        if temp_grad_2_path:
            os.remove(temp_grad_2_path)
        if temp_lime_2_path:
            os.remove(temp_lime_2_path)
        if temp_shap_2_path:
            os.remove(temp_shap_2_path)

        return response



def modelInfo(request):
    return render(request, 'modelInfo.html')

def userGuide(request):
    return render(request, 'userGuide.html')
