## django web개발 관련 파일
* 프로젝트 이름: apple
* 앱 이름: te
* 앱 위치: project/apple/te
* 모델 api: apple/api 
### views: api 요청 처리  
* webAPI + webPredictImage: web에서 들어온 요청 처리(return 결과값 dictionary)  
* mobileAPI + mobilePredictImage: mobile에서 들어온 요청 처리 (return 결과값 json)  
* get_result: 모델 적용 + 처리된 이미지 저장 + 결과 return(예측된 class_id, score 등 dictionary)
### urls: 요청 url과 view 맵핑
### serializers: 업로드된 데이터 형태 변환
### ETC
* 앱에 적용한 모델: te/model_d -> model.h5 : 현재 적용한 모델의 weights파일
* client로 부터 받은 파일: apple/media -> 년도/월/일upload 형태로 저장
* prediction후의 이미지 파일: te/static/results에 저장
* DB(SQLite): apple/db.sqlite3 -> 여기에 질병, 농약 정보 저장되어있음
* front-end 부분: te/templates/devfo/index.html
