import os
import django
import csv
import sys

#현재 디렉토리 경로 표시
os.chdir(".")
print("Current dir=", end=""), print(os.getcwd())

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("BASE_DIR=", end=""), print(BASE_DIR)

sys.path.append(BASE_DIR)

# 프로젝트명.settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "apple.settings")	# 1. 여기서 프로젝트명.settings입력
django.setup()

from te.models import Pest

# CSV 파일 경로
CSV_PATH = './te/static/applepest1.csv'

#encoding 설정 필요

with open(CSV_PATH, newline='', encoding='utf-8-sig') as csvfile:
    data_reader = csv.DictReader(csvfile)

    for row in data_reader:
        #print(row)
        Pest.objects.create(
            pestiKorName = row['pestiKorName'],
            regCpntQnty = row['regCpntQnty'],
            pestiBrandName = row['pestiBrandName'],
            compName = row['compName'],
            useName = row['useName'],
            indictSymbl = row['indictSymbl'],
            cropName = row['cropName'],
            diseaseWeedName = row['diseaseWeedName'], 
            pestiUse = row['pestiUse'], 
            dilutUnit = row['dilutUnit'],
            useSuittime = row['useSuittime'],
            useNum = row['useNum']
        )
