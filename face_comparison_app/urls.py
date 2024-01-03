from django.urls import path
from .views import compare_images_api

urlpatterns = [
    path('compare_images/', compare_images_api, name='compare_images_api'),
]
