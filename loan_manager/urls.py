from django.urls import path
from . import views

urlpatterns = [
    path('', views.request_list, name='request_list'),
    path('add/', views.request_form, name='request_add'),
    path('eda/', views.eda_view, name='eda'),
    path('performance/', views.performance_view, name='performance'),
    path('delete/<int:pk>/', views.request_delete, name='request_delete'),
    path('override/<int:pk>/', views.request_override, name='request_override'),
] 