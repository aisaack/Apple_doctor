from rest_framework import serializers
from te.models import upload_image

class upload_imageSerializers(serializers.ModelSerializer):
    class Meta:
        model = upload_image
        fields = '__all__'