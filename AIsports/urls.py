
from django.contrib import admin
from django.urls import path
import home.views
from django.conf.urls import include, url



urlpatterns = [
    path('admin/', admin.site.urls),
    path('main/',home.views.main,name='main'),
    path('',home.views.test1,name='test1'),
    path('predict/',home.views.predict,name='predict'),
    path('findmatch/', home.views.findmatch, name='findmatch'),
    path('result/', home.views.result, name='result'),
    path('chat/', include('chat.urls')),
    path('member/',home.views.member, name='member'),
    
    
]