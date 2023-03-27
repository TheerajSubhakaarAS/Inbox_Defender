



from django.shortcuts import render
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.views.decorators.cache import cache_control
import os


model1 = pickle.load(open('C:/Users/Dhanush/AppData/Local/Programs/Python/Python310/Scripts/random.sav','rb'))
model2 = pickle.load(open('C:/Users/Dhanush/AppData/Local/Programs/Python/Python310/Scripts/SVC.sav','rb'))
model3 = pickle.load(open('C:/Users/Dhanush/AppData/Local/Programs/Python/Python310/Scripts/log.sav','rb'))
# Create your views here.
@cache_control(no_cache=True, must_revalidate=True, no_store=True)
def index(request):
      return render(request, 'index.html')
       
@cache_control(no_cache=True, must_revalidate=True, no_store=True)
def checkSpam(request):
    if(request.method == "POST"):
            algo = request.POST.get("algo")
            rawData = request.POST.get("rawdata")
            if(algo == "RANDOM CLASSIFIER"):
                return render(request, 'output.html', {"answer" : model1.predict([rawData])[0]})
            elif(algo == "SUPPORT VECTOR MACHINE"):
                return render(request, 'output.html', {"answer" : model2.predict([rawData])[0]})
            elif(algo == "LOGISTIC REGRESSION"):
                return render(request, 'output.html', {"answer" : model3.predict([rawData])[0]})
    else:
        return render(request, 'index.html')


