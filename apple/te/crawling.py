import requests
import xmltodict
import time

#농약상세정보 api 사용

def agchmParsing():

    #public api settings
    cropName = "사과" #작물명 검색
    key = "202049c1a90f95ff7c2f353ff3aac72145b7"

    url = "http://pis.rda.go.kr/openApi/service.do?apiKey=" + key + "&serviceCode=SVC01" + "&cropName=" + cropName

    req = requests.get(url).content
    # xmltodict : It changes xml type into dictionary.
    xmlObject = xmltodict.parse(req)
    allData = xmlObject['service']['list']['item']

    return allData