from django.shortcuts import render, redirect
from django.http import HttpResponse
from .forms import ImageUploadForm
import os

from django.core.files.storage import default_storage
import base64

from django.conf import settings
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
from ultralytics import YOLO
import io

from django.core.mail import send_mail
from django.core.mail import EmailMessage

model_path = os.path.join(settings.BASE_DIR, 'models/yolo12x.pt')
model = YOLO(model_path)

model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def encode_img(img):
    output_io = io.BytesIO()
    img.save(output_io, format='JPEG')
    output_io.seek(0)

    encoded_string = base64.b64encode(output_io.read()).decode('utf-8')
    mime_type = "image/jpg"
    image_url = f"data:{mime_type};base64,{encoded_string}"
    return image_url

def image_processing(img):
    preprocess = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) неверно считается норма
    ])
    img_tensor = preprocess(img).unsqueeze(0)
    num_person = 0
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        results = model(img_tensor)
        boxes = results[0].boxes
        if boxes:
            xyxy = boxes.xyxy
            conf = boxes.conf
            cls = boxes.cls
            draw = ImageDraw.Draw(img)

            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i].cpu().numpy().astype(int)
                confidence = conf[i].cpu().numpy()
                class_id = int(cls[i].cpu().numpy())
                class_name = results[0].names[class_id]

                if class_name!="person":
                    continue
                num_person+=1
                label = f"{class_name} {confidence:.2f}"

                draw.rectangle(((x1, y1), (x2, y2)), outline="red", width=2)
                draw.text((x1, y1 - 15), label, fill="red", font_size=15)
            image_url = encode_img(img)
        else:
            image_url = encode_img(img)
    return image_url, num_person

def send_mesg(image_url, num_person):
    subject = f"Обнаружено {num_person} человека на изображении"
    body = f"При загрузке изображения найдено {num_person} человека. Смотрите вложение."
    to = [settings.EMAIL_ADMIN]

    try:
        header, b64 = image_url.split(',', 1)
    except ValueError:
        b64 = image_url
    image_bytes = base64.b64decode(b64)

    email = EmailMessage(
        subject=subject,
        body=body,
        from_email=settings.DEFAULT_FROM_EMAIL,
        to=to,
    )
    email.attach(
        filename="processed_image.jpg",
        content=image_bytes,
        mimetype="image/jpeg"
    )
    email.send(fail_silently=False)



def image_upload_view(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = request.FILES['image']

            try:
                img = Image.open(image).convert('RGB').resize((640, 640))
            except Exception as e:
                return render(
                    request,
                    'cv_app/upload_form.html',
                    {'form': form, 'error_message': f"Ошибка открытия изображения: {e}"}
                )

            image_url, num_person = image_processing(img)

            if num_person <= 2 and settings.EMAIL_ADMIN:#число 2 обосновано
                send_mesg(image_url, num_person)
            context = {
                    'form': form,
                    'image_url': image_url,
                    'num_person': num_person,
            }
            return render(request, 'cv_app/results.html', context)

        return render(request, 'cv_app/upload_form.html', {'form': form, 'errors': form.errors})

    else:
        form = ImageUploadForm()
    return render(request, 'cv_app/upload_form.html', {'form': form})
