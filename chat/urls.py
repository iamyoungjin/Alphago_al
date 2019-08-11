# chat/urls.py
from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^$', views.index1, name='index1'),
    url(r'^(?P<room_name>[^/]+)/$', views.room, name='room'),
]