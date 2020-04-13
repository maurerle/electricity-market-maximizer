from django.db import models
from django.contrib.auth.models import User
# Create your models here.
"""
class Operator(models.Model):
    # Define the operator sql Table and the attributes
    # of each entry
    name = models.CharField(max_length=200, null=True)
    email = models.CharField(max_length=200, null=True)
    operator = models.CharField(max_length=200, null=True)
    date_created = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        # Return the operator name in the admin page
        return self.name
"""
class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    operator = models.CharField(max_length=200)

    def __str__(self):
        return self.user.username
