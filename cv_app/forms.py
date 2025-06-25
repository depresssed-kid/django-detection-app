from django import forms
# from .models import ImageUpload # Раскомментируйте, если хотите хранить информацию о загрузках

class ImageUploadForm(forms.Form):
    image = forms.ImageField(label='Загрузите изображение')