# Generated by Django 3.1.2 on 2020-10-16 07:53

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('te', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='upload_image',
            name='name',
            field=models.CharField(default='default_name', max_length=150),
        ),
    ]
