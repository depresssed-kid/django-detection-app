from django.shortcuts import render, redirect
from django.http import HttpResponse
from .forms import ImageUploadForm
import os

from django.core.files.storage import default_storage
import base64

from django.conf import settings  # Импортируйте settings
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
from ultralytics import YOLO
import io
#model_path = os.path.join(settings.BASE_DIR, 'models/yolo12x.pt')
#model = torch.load(model_path, weights_only=False)

model_path = os.path.join(settings.BASE_DIR, 'models/yolo12x.pt')
model = YOLO(model_path)

model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def image_upload_view(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = request.FILES['image']

            try:
                img = Image.open(image).convert('RGB')
            except Exception as e:
                return render(request, 'cv_app/upload_form.html', {'form': form, 'error_message': f"Ошибка открытия изображения: {e}"})

            preprocess = transforms.Compose([
                transforms.Resize((640, 640)),
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) неверно считается норма
            ])
            img_tensor = preprocess(img).unsqueeze(0)

            with torch.no_grad():
                img_tensor = img_tensor.to(device)
                results = model(img_tensor)
                boxes = results[0].boxes
                if boxes:
                    xyxy = boxes.xyxy
                    conf = boxes.conf
                    cls = boxes.cls

                    img = Image.open(image).convert("RGB").resize((640, 640))
                    draw = ImageDraw.Draw(img)

                    for i in range(len(xyxy)):
                        x1, y1, x2, y2 = xyxy[i].cpu().numpy().astype(int)
                        confidence = conf[i].cpu().numpy()
                        class_id = int(cls[i].cpu().numpy())
                        class_name = results[0].names[class_id]
                        label = f"{class_name} {confidence:.2f}"

                        draw.rectangle(((x1, y1), (x2, y2)), outline="red", width=2)
                        draw.text((x1, y1 - 15), label, fill="red", font_size=15)

                    output_io = io.BytesIO()
                    img.save(output_io, format='JPEG')
                    output_io.seek(0)


                    encoded_string = base64.b64encode(output_io.read()).decode('utf-8')
                    mime_type = "image/jpeg"
                    image_url = f"data:{mime_type};base64,{encoded_string}"
                else:
                    encoded_string = base64.b64encode(image.read()).decode('utf-8')
                    mime_type = "image/jpeg"
                    image_url = f"data:{mime_type};base64,{encoded_string}"


            # 4. Передача результата в шаблон:
            # context = {'form': form, 'predicted_class': predicted_class, 'image_url': image_upload.image.url if 'image_upload' in locals() else None}
            context = {'form': form,  'image_url': image_url}
            return render(request, 'cv_app/results.html', context)
            # return redirect('results') #  Перенаправьте на страницу с результатами
        else:
            return render(request, 'cv_app/upload_form.html', {'form': form, 'errors': form.errors})

    else:
        form = ImageUploadForm()
    return render(request, 'cv_app/upload_form.html', {'form': form})

#  (Если хотите отдельную вьюху для результатов)
# def results_view(request):
#     context = {
#         #  Данные с прошлого запроса (например, predicted_class, image_url)
#     }
#     return render(request, 'cv_app/results.html', context)
