from django.contrib import admin

from te.models import upload_image
from te.models import Category, Disease
from te.models import Pest


admin.site.register(upload_image) # 기본 ModelAdmin

class CategoryAdmin(admin.ModelAdmin):
    list_display = ['name', 'slug']
    prepopulated_fields = {'slug':('name',)}

admin.site.register(Category, CategoryAdmin)

class DiseaseAdmin(admin.ModelAdmin):
    list_display=['name','description','category','symptom','pest']
    list_editable=['description', 'symptom','pest']
    prepopulated_fields = {'slug':('name',)}
    list_per_page = 10

admin.site.register(Disease, DiseaseAdmin)


#0   pestiKorName  1   regCpntQnty  2   pestiBrandName  3   compName  4   toxicName 
# 5 fishToxicGubun  6   useName  7   indictSymbl  8   cropName   9   diseaseWeedName 
#10  pestiUse  11  dilutUnit   12  useSuittime  13  useNum 

admin.site.register(Pest)
