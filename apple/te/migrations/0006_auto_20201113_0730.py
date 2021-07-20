# Generated by Django 3.1.2 on 2020-11-13 07:30

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('te', '0005_pest'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='pest',
            name='fishToxicGubun',
        ),
        migrations.RemoveField(
            model_name='pest',
            name='toxicName',
        ),
        migrations.AddField(
            model_name='pest',
            name='slug',
            field=models.SlugField(default=None, max_length=250, unique=True),
        ),
    ]
