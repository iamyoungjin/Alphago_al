from django.db import models

# Create your models here.

class Match(models.Model):
    year = models.IntegerField()
    month = models.IntegerField()
    day = models.IntegerField()
    updated_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return str(self.year) + str(self.month) + str(self.day)
        
class Detail(models.Model):
    text = models.TextField()
    time = models.ForeignKey(Match, on_delete=models.CASCADE)