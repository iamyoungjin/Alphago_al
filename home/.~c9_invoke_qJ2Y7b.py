from django.shortcuts import render
import pickle 
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from .models import Match

HST = [15.94, 7.54, 3.24, 5.78, 0.6, 5.56, 2.28, 0.4, 0.86, 2.1, 0.82, 3.58, 15.48, 0.28, 3.02, 1.08, 11.58, 7.42, 0.52, 2.14, 1.02, 2.28, 4.1, 14.32, 13.72, 15.28, 0.72, 4.26, 9.54, 2.5, 3.68, 1.2, 1.42, 0.48, 6.48, 4.96, 5.76, 3.58, 12.92, 2.34, 5.6, 8.5, 3.38, 2.26]
AST = [11.8, 6.82, 2.22, 4.58, 0.5, 4.34, 1.74, 0.2, 0.52, 1.7, 0.5, 2.44, 11.18, 0.44, 3.1, 0.7, 7.74, 4.66, 0.48, 1.48, 0.92, 1.96, 3.6, 10.98, 10.14, 12.02, 0.24, 3.02, 6.7, 1.54, 2.16, 1.1, 1.3, 0.16, 4.6, 3.0, 4.6, 2.54, 9.48, 1.76, 3.9, 6.22, 2.94, 1.8]
HAS = [1.4456738619999998, 0.6838382009999999, 0.293850898, 0.524215491, 0.054416833, 0.504262652, 0.206783965, 0.036277889, 0.07799746099999999, 0.190458915, 0.074369672, 0.32468710300000003, 1.40395429, 0.025394522000000003, 0.27389805899999997, 0.097950299, 1.050244876, 0.6729548340000001, 0.047161255, 0.194086704, 0.092508616, 0.206783965, 0.371848358, 1.298748413, 1.24433158, 1.385815346, 0.06530019999999999, 0.38635951399999996, 0.8652276440000001, 0.226736804, 0.333756575, 0.108833666, 0.128786505, 0.04353346599999999, 0.587701796, 0.449845819, 0.522401596, 0.32468710300000003, 1.171775803, 0.21222564800000002, 0.507890441, 0.770905133, 0.306548159, 0.204970071]
HDS = [0.7447198140000001, 0.83262117, 0.34183860299999996, 0.58356733, 0.090343059, 0.63484312, 0.288121109, 0.070809425, 0.12941032800000002, 0.293004517, 0.178244415, 0.42485655, 0.688560615, 0.05615919900000001, 0.40776462, 0.22463679600000003, 0.935172751, 0.817970944, 0.13673544099999999, 0.354047125, 0.09522646800000001, 0.239287022, 0.46880722700000005, 0.669026981, 0.774020266, 0.637284825, 0.051275790999999994, 0.512757905, 0.9327310459999999, 0.31742156, 0.388230985, 0.188011232, 0.19045293600000002, 0.051275790999999994, 0.644609938, 0.520083018, 0.820412648, 0.41508973299999996, 0.8985471859999999, 0.322304969, 0.786228788, 0.954706385, 0.5249664270000001, 0.36381394200000006]
AAS = [1.4406055430000002, 0.83262117, 0.271029178, 0.559150287, 0.061042608, 0.529849835, 0.212428275, 0.024417043, 0.063484312, 0.20754486600000002, 0.061042608, 0.297887926, 1.3649127090000002, 0.053717495, 0.378464168, 0.085459651, 0.944939568, 0.568917104, 0.058600902999999996, 0.18068611899999998, 0.11231839800000001, 0.239287022, 0.439506776, 1.340495666, 1.237944085, 1.46746429, 0.029300452, 0.368697351, 0.817970944, 0.188011232, 0.263704065, 0.134293737, 0.15871078, 0.019533633999999998, 0.561591991, 0.366255646, 0.561591991, 0.310096447, 1.157367843, 0.214869979, 0.47613234, 0.75937004, 0.35893053399999997, 0.219753388]
ADS = [0.785416289, 0.903319427, 0.399056775, 0.640304734, 0.074369672, 0.640304734, 0.26664248100000004, 0.074369672, 0.11064756, 0.321059314, 0.12697261, 0.37729004200000005, 0.636676945, 0.072555777, 0.39180119700000005, 0.214039543, 0.914202793, 0.803555233, 0.14148376599999998, 0.322873209, 0.11971703199999999, 0.21222564800000002, 0.448031925, 0.785416289, 0.7872301829999999, 0.6910937779999999, 0.041719572, 0.449845819, 0.972247415, 0.362778886, 0.400870669, 0.221295121, 0.19590059899999998, 0.061672410999999996, 0.6203518960000001, 0.565935063, 0.828949755, 0.38635951399999996, 0.92508616, 0.304734264, 0.678396517, 0.9196444770000001, 0.484309813, 0.31924542]

model = XGBClassifier()


# Create your views here.
def main(request):
    options = ['Arsenal','Aston Villa','Birmingham','Blackburn','Blackpool','Bolton','Bournemouth','Bradford','Brighton','Burnley',
    'Cardiff','Charlton','Chelsea','Coventry','Crystal Palace','Derby','Everton','Fulham','Huddersfield','Hull','Ipswich','Leeds',
    'Leicester','Liverpool','Man City','Man United','Middlesboro','Middlesbrough','Newcastle',
    'Norwich','Portsmouth','QPR','Reading','Sheffield United','Southampton','Stoke','Sunderland','Swansea',
    'Tottenham','Watford','West Brom','West Ham','Wigan','Wolves']
    return render(request,'main.html',{'options':options})
    
def index(request):
    return render(request,'index.html')
    
def predict(request):
    if request.method == 'GET':
        
        
        
        filename = 'finalized_model.sav'
        model = pickle.load(open(filename, 'rb'))

        home = request.GET['home_team']
        away = request.GET['away_team']
        
        result_input = pd.DataFrame(columns=('HST','AST','HAS','HDS','AAS','ADS',))
        result_input.loc[0]=np.nan
        
        result_input['HST'] = 10
        result_input['AST'] = 6
        result_input['HAS'] = 1.5
        result_input['HDS'] = 0.9
        result_input['AAS'] = 0.3
        result_input['ADS'] = 0.9
        
        result = model.predict(result_input)
        if home==away:
            result = '같은 팀을 선택하셨습니다! 다시 입력해주세요!'
        elif result == 1:
            result = '{}팀이 이깁니다!'.format(home)
        elif result == 0:
            result = '무승부입니다!'
        else:
            result = '{}팀이 이깁니다!'.format(away)
        return render(request, 'predict.html' , {'result': result })
    else:
        return render(request, 'main.html')

    
a = 0
def findmatch(request):
    if a == 0:
        driver = webdriver.Chrome('/Users/user/Downloads/chromedriver_win32/chromedriver.exe')
        driver.implicitly_wait(3)
    
        for i in range(8,13):
            if i < 10:
                driver.get('https://sports.news.naver.com/wfootball/schedule/index.nhn?year=2019&month=0{}&category=premier'.format(i))
            else:
                driver.get('https://sports.news.naver.com/wfootball/schedule/index.nhn?year=2019&month={}&category=premier'.format(i))
            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            notices = soup.select('#_monthlyScheduleList > tr')
        
            for notice in notices:
                lst = notice.text.split('\n')
                if '경기가 없습니다.' in lst:
                    mat = Match()
                    mat.year = 2019
                    mat.month = lst[3][0]
                    mat.day = int(lst[3][2] + lst[3][3])
                    mat.content = lst[8]
                    # print(lst[3])   #날짜
                    # print(lst[8])   #경기내용
                    # print('')
                elif ('경기가 없습니다.' not in lst):
                    mat = Match()
                    if ('\t\t' in lst):
                        mat = Match()
                        mat.year = 2019
                        mat.month = lst[3][0]
                        mat.day = int(lst[3][2] + lst[3][3])
                        mat.content = lst[8] + lst[9] + '   ' + lst[16]+'vs'+lst[20]
                        # print(lst[3])   #날짜
                        # print(lst[8], lst[9], lst[16], lst[20]) #경기내용
                        # print('')
                    else:
                        print(lst[3], lst[4], lst[11], lst[15]) #경기내용
                        print('')
                else:
                    pass
        
        for i in range(1,6):
            driver.get('https://sports.news.naver.com/wfootball/schedule/index.nhn?year=2020&month=0{}&category=premier'.format(i))
            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            notices = soup.select('#_monthlyScheduleList > tr')
        
            for notice in notices:
                lst = notice.text.split('\n')
                if '경기가 없습니다.' in lst:
                    print(lst[3])   #날짜
                    print(lst[8])   #경기내용
                    print('')
                elif ('경기가 없습니다.' not in lst) and ('\t\t' in lst):
                    print(lst[3])   #날짜
                    print(lst[8], lst[9], lst[16], lst[20]) #경기내용
                    print('')
                elif ('경기가 없습니다.' not in lst) and ('\t\t' not in lst):
                    print(lst[3], lst[4], lst[11], lst[15]) #경기내용
                    print('')
                else:
                    pass
        a+=1
        return render(request, 'findmatch.html')
    else:
        
        return render(request, 'findmatch.html')