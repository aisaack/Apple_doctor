from django.http.response import JsonResponse
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage, default_storage
from te.views import upload_image
from .serializers import upload_imageSerializers
from django.http import Http404
from django.core.files.base import ContentFile
from django.conf import settings

# 3rd party improts
from rest_framework.views import APIView
from rest_framework.response import Response

#=====visualization
import colorsys
import cv2
import os
import sys
from skimage.measure import find_contours
import scipy
import numpy as np
import random
import itertools
import matplotlib.pyplot as plt
from matplotlib import patches, lines
from matplotlib.patches import Polygon,Rectangle
import json
from tensorflow import Graph, Session
import skimage.io
import tensorflow as tf
from te.model_d.mrcnn import model as modellib,utils
import te.model_d.apple as apple
from te.model_d.mrcnn import visualize
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model



global session
global graph

class webAPI(APIView):
    def post(self, request, *args, **kwargs):
        serializer = upload_imageSerializers(data = request.data)
        print(request.data)
        print('=====webAPI2 OKAYYYYYYYYYYYYYYY')
        if serializer.is_valid():
            print('serializer is validated')        
            serializer.save()
            ret = webPredictImage(request, serializer.data)
            return Response(ret)
            #return Response(serializer.data)
        return Response(serializer.errors)


class mobileAPI(APIView):
    def post(self, request, *args, **kwargs):
        print('=====mobileAPI OKAYYYYYYYYYYYYYYY')
        return mobilePredictImage(request)
        
class getAPI(APIView):
    def get_object(self, pk): 
        try:
            return upload_image.objects.get(pk=pk)
        except upload_image.DoesNotExist:
            raise Http404

    def get(self, request, pk, format=None, *args, **kwargs):
        upload_image = self.get_object(pk)
        serializer = upload_imageSerializers(upload_image)
        return Response(serializer.data)

class InferenceConfig(apple.AppleConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

# config = InferenceConfig()
# model_graph = Graph()

config = InferenceConfig()
session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
graph = tf.get_default_graph()

set_session(session)
MODEL_PATH = os.path.abspath("te/model_d/mrcnn/model.py")
WIEGHT_PATH = os.path.abspath("te/model_d/model.h5")
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_PATH, config=config)
model.load_weights(WIEGHT_PATH, by_name=True)
model.keras_model._make_predict_function()


#나중에 영어 -> 한글로 바꾸기
# class_names = ['Nothing','sooty-blotch', 'bitter-rot','brown-rot','mar-blotch','white-rot']
class_names =['Nothing','Bitter rot','brown rot','Mar blotch','White rot', 'Normal','Sooty/Fly']
real_names=['알 수 없는 병','탄저병(Bitter rot)','잿빛무늬병(Brown rot)','갈반병(Mar blotch)','겹무늬썩음병(White rot)','정상','그을음병(Sooty/fly)']

def get_result(f_path):
    pred = None
    print(f_path)
    with graph.as_default():
        x = skimage.io.imread('./'+f_path)
        set_session(session)
        # model = modellib.MaskRCNN(mode="inference", model_dir='te/model_d', config=config)
        # model.load_weights("te/model_d/model.h5", by_name=True)      
        # model.keras_model._make_predict_function()
        predi = model.detect([x], verbose=0)
        r= predi[0]
        save_d = 'te/static/results/'
        image_name = f_path.split('/')[-1]
        visualize.display_instances(x, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'],save_dir=save_d + image_name)

    # with model_graph.as_default():
    #     tf_session = Session()
    #     with tf_session.as_default():
    #         x = skimage.io.imread('./'+f_path)
            
    #         model = modellib.MaskRCNN(mode="inference", model_dir='te/model_d', config=config)
    #         model.load_weights("te/model_d/model.h5", by_name=True)      
    #         model.keras_model._make_predict_function()
    #         predi = TeConfig.model.detect([x], verbose=0)
    #         r= predi[0]
    #         save_d = 'te/static/results/'
    #         image_name = f_path.split('/')[-1]
    #         visualize.display_instances(x, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'],save_dir=save_d + image_name)
    return predi[0]

def webPredictImage(request, data):
    print(request)
    print("===============Image2 data:", data)
    filePathName = data['photo']
    pred = get_result(filePathName)
    labels = list([real_names[m],class_names[m]] for m in pred['class_ids']) if len(pred['class_ids']) > 0 else [[real_names[0],'']]
    scores = list(round(s*100,2) for s in pred['scores']) if len(pred['scores']) > 0 else [100]
    context = {'filePathName': filePathName,
    'resultZip':zip(labels, scores),
    'resultDir': f"../../../static/results/{filePathName.split('/')[-1]}"}
    return context
    # return render(request, 'devfo/index.html', context)

def mobilePredictImage(request):
    print("=====mobile")
    file = request.FILES['file']    
    path = default_storage.save(str(file)+".jpg", ContentFile(file.read()))
    file_path = os.path.join(settings.MEDIA_ROOT, path)

    pred = get_result(file_path[10:])
    labels = list([real_names[m],class_names[m]] for m in pred['class_ids']) if len(pred['class_ids']) > 0 else [[real_names[0],'']]
    scores = list(round(s*100,2) for s in pred['scores']) if len(pred['scores']) > 0 else [100]
    
    print(type(labels[0][0]))
    print(type(scores[0]))
    context = {
        'disease' : labels[0][0],
        'per' : scores[0]
    }
    print(context)
    return JsonResponse(context, status=200) #json 형식으로


    