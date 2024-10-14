from django.db import models


class Person(models.Model):
    name = models.CharField(max_length=255, blank=True, null=True)
    label_name = models.CharField(max_length=255)
    birth_year = models.DateField()
    bio = models.TextField()
    occupation = models.CharField(max_length=255)
    image = models.FileField(upload_to="persons", null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name
