import pytest
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
from PIL import Image
from io import BytesIO
import base64
import PyPDF2

@pytest.fixture
def uploaded_image():
    img = Image.new('RGB', (224,224), color='red')
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return SimpleUploadedFile("test_image.png", img_byte_arr.read(), content_type='image/png')

@pytest.fixture
def uploaded_wide_image():
    img = Image.new('RGB', (640,161), color='red')
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return SimpleUploadedFile("test_image.png", img_byte_arr.read(), content_type='image/png')

@pytest.mark.django_db
def test_main_page_post_valid(client, uploaded_image):
    url = reverse('main')
    form_data = {'image': uploaded_image}
    response = client.post(url, data=form_data)

    assert response.status_code == 200
    assert 'prediction_1' in response.context
    assert 'label_1' in response.context
    assert 'image_1' in response.context

@pytest.mark.django_db
def test_main_page_post_wide_valid(client, uploaded_wide_image):
    url = reverse('main')
    form_data = {'image': uploaded_wide_image}
    response = client.post(url, data=form_data)

    assert response.status_code == 200
    assert 'prediction_1' in response.context
    assert 'label_1' in response.context
    assert 'image_1' in response.context
    assert 'prediction_2' in response.context
    assert 'label_2' in response.context
    assert 'image_2' in response.context

@pytest.mark.django_db
def test_main_page_post_invalid(client):
    url = reverse('main')
    data = {}
    response = client.post(url, data)

    assert response.status_code == 200
    assert 'form' in response.context
    assert len(response.context['form'].errors) > 0

@pytest.fixture
def invalid_image():
    img_byte_arr = BytesIO(b"This is not an image")
    img_byte_arr.seek(0)
    return SimpleUploadedFile("invalid_file.txt", img_byte_arr.read(), content_type='text/plain')

@pytest.mark.django_db
def test_main_page_post_invalid_image(client, invalid_image):
    url = reverse('main')
    data = {'image': invalid_image}
    response = client.post(url, data)

    assert response.status_code == 200
    assert 'form' in response.context
    assert len(response.context['form'].errors) > 0

@pytest.mark.django_db
def test_generate_pdf_post(client, uploaded_image):
    url = reverse('generate_pdf')
    image_data_1 = base64.b64encode(uploaded_image.read()).decode('utf-8')

    data = {
        'prediction_1': '0.95',
        'label_1': '1',
        'image_1': image_data_1,
        'prediction_2': '',
        'label_2': '',
        'image_2': '',
    }

    response = client.post(url, data)

    assert response.status_code == 200
    assert response['Content-Type'] == 'application/pdf'

    pdf_reader = PyPDF2.PdfReader(BytesIO(response.content))
    pdf_text = ''
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        pdf_text += page.extract_text()

    assert 'Prediction Report' in pdf_text
    assert 'Knee is unhealthy' in pdf_text

@pytest.mark.django_db
def test_generate_pdf_post_wide(client, uploaded_image):
    url = reverse('generate_pdf')
    image_data_1 = base64.b64encode(uploaded_image.read()).decode('utf-8')

    data = {
        'prediction_1': '0.14',
        'label_1': '0',
        'image_1': image_data_1,
        'prediction_2': '0.99',
        'label_2': '1',
        'image_2': image_data_1,
    }

    response = client.post(url, data)

    assert response.status_code == 200
    assert response['Content-Type'] == 'application/pdf'

    pdf_reader = PyPDF2.PdfReader(BytesIO(response.content))
    pdf_text = ''
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        pdf_text += page.extract_text()

    assert 'Prediction Report' in pdf_text
    assert 'Left knee is healthy' in pdf_text
    assert 'Right knee is unhealthy' in pdf_text

def test_model_info(client):
    url = reverse('modelInfo')
    response = client.get(url)
    assert response.status_code == 200
    assert b'Model Info' in response.content

def test_user_guide(client):
    url = reverse('userGuide')
    response = client.get(url)
    assert response.status_code == 200
    assert b'User Guide' in response.content
