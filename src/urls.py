from django.urls import path, include

from src import views

urlpatterns = [
    path('', views.IndexView.as_view(), name="index")
    # path('predict/', views.PredictView.as_view())
]
