
from django.urls import path
from django.urls import include
from prediction import views
urlpatterns = [
    path('prediction/',include('prediction.urls'))

]
