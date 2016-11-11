# -*- coding: utf-8 -*-

from pandas import read_csv, DataFrame, Series
from sklearn import cross_validation, svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import math

def stat(true, pred, regression_name):
    print 'Report || ' + regression_name
    print 'Средняя абсолютная ошибка: %5f' % mean_absolute_error(true, pred)
    print 'R^2 score: %5f' % r2_score(true, pred)
    print 'Истинные значения     ',true
    print 'Предсказанные значения', pred
    print '***' * 5
    print ' '


data = read_csv('DATA.csv', sep=';', header=0)
# header = read_csv('pacient-header.csv', sep=';', header=None)

label = LabelEncoder()
encode_dict = {}

label.fit(data['Diagnoz osnovnoj'].drop_duplicates()) #задаем список значений для кодирования
encode_dict['diagnoz'] = list(label.classes_)
data['Diagnoz osnovnoj'] = label.transform(data['Diagnoz osnovnoj']) #заменяем значения из списка кодами закодированных элементов

label.fit(data['Operirujushhij hirurg'].drop_duplicates())
encode_dict['hirurg'] = list(label.classes_)
data['Operirujushhij hirurg'] = label.transform(data['Operirujushhij hirurg'])

label.fit(data['Vozrast'].drop_duplicates())
encode_dict['Vozrast'] = list(label.classes_)
data['Vozrast'] = label.transform(data['Vozrast'])

label.fit(data["Vysejannaja kul'tura"].drop_duplicates())
encode_dict['bakteriya'] = list(label.classes_)
for k in range(0, len(encode_dict['bakteriya'])):
    if isinstance(encode_dict['bakteriya'][k], float) and math.isnan(encode_dict['bakteriya'][k]):
        encode_dict['bakteriya'][k] = 'none'
data["Vysejannaja kul'tura"] = label.transform(data["Vysejannaja kul'tura"])

label.fit(data["Ajerobnost'"].drop_duplicates())
encode_dict["Ajerobnost'"] = list(label.classes_)
data["Ajerobnost'"] = label.transform(data["Ajerobnost'"])

data.to_csv('DATA_encode.csv', sep=';')
for k, v in encode_dict.iteritems():
    print k + ":[" + ", ".join(v) + "]"

# data.pivot_table('№', ['Operirujushhij hirurg'], "Ajerobnost'", 'count').fillna(0).plot(kind='bar', stacked=True).get_figure().savefig('plots/hirurg_aerob.png')
# data.pivot_table('№', ['Vremja operacii'], "Ajerobnost'", 'count').fillna(0).plot().get_figure().savefig('plots/vremya_aerob.png')
# data.pivot_table('№', ['Vozrast'], "Ajerobnost'", 'count').fillna(0).plot(kind='bar', stacked=True).get_figure().savefig('plots/vosrast_aerob.png')
# data.pivot_table('№', ['Mesjac operacii'], "Ajerobnost'", 'count').fillna(0).plot(kind='bar', stacked=True).get_figure().savefig('plots/mesyac_aerob.png')

# data.pivot_table('№', ['Operirujushhij hirurg'], "Rezul'tat1", 'count').fillna(0).plot(kind='bar', stacked=True).get_figure().savefig('plots/hirurg_res1.png')
# data.pivot_table('№', ['Vremja operacii'], "Rezul'tat1", 'count').fillna(0).plot().get_figure().savefig('plots/vremya_res1.png')
# data.pivot_table('№', ['Vozrast'], "Rezul'tat1", 'count').fillna(0).plot(kind='bar', stacked=True).get_figure().savefig('plots/vosrast_res1.png')
# data.pivot_table('№', ['Mesjac operacii'], "Rezul'tat1", 'count').fillna(0).plot(kind='bar', stacked=True).get_figure().savefig('plots/mesyac_res1.png')
# data.pivot_table('№', ["Vysejannaja kul'tura"], "Rezul'tat1", 'count').fillna(0).plot(kind='bar', stacked=True).get_figure().savefig('plots/vysev_res1.png')
# data.pivot_table('№', ['Vremya prebivaniya v stacionare (dni)'], "Rezul'tat1", 'count').fillna(0).plot(kind='bar', stacked=True).get_figure().savefig('plots/stacionar_res1.png')

# data.pivot_table('PassengerId', ['Parch'], 'Survived', 'count').plot(ax=axes[1], title='Parch')
plt.show()

data = data.drop(['№', "Bol'noj", "Nomer istorii bolezni", "Data operacii", "Chuvstvitel'nost' k ab", "Ustojchivost' k a/b"], axis=1)

train = data.drop(["Ajerobnost'", "Rezul'tat1", "Vremya prebivaniya v stacionare (dni)"], axis=1) #DROP TARGET
target = data["Vremya prebivaniya v stacionare (dni)"]


kfold = 10 #количество подвыборок для валидации
itog_val = {} #список для записи результатов кросс валидации разных алгоритмов
models = {}
getlist = []
name_groups = ''

# train = data.iloc[:, 2:]
models['RandomForestRegression'] = RandomForestRegressor(n_estimators=1000) #в параметре передаем кол-во деревьев
models['ExtraTreesRegression'] = ExtraTreesRegressor(n_estimators=1000)
models['KNeighborsRegression'] = KNeighborsRegressor(n_neighbors=20) #в параметре передаем кол-во соседей
models['LogisticRegression'] = LogisticRegression(penalty='l2', tol=0.01)
models['SVC'] = svm.SVC() #по умолчанию kernek='rbf'
models['SVC'].probability = True
# models['OneVsRestClassifier'] = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
#                                  random_state=np.random.RandomState(0)))

for name, model in models.items():
    scores = cross_validation.cross_val_score(model, train, target, cv=kfold)
    itog_val[name] = scores.mean()
print 'Кросс-валидация:'
print itog_val
print ''

ROCtrainTRN, ROCtestTRN, ROCtrainTRG, ROCtestTRG = cross_validation.train_test_split(train, target, test_size=0.4)

for name, model in models.items():
    fit = model.fit(ROCtrainTRN, ROCtrainTRG)
    pred = fit.predict(ROCtestTRN)
    stat(ROCtestTRG, pred, name)

feature_importance = models['RandomForestRegression'].feature_importances_
names_feature = list(train.columns.values)
f_i_zipped = zip(names_feature, feature_importance.tolist())
f_i_zipped.sort(key = lambda t:t[1], reverse=True)
print 'Влияние факторов [top 10], %:'
for n, f in f_i_zipped[:10]:
    print " %10f - %s" % (f, n)
print 'max: ' + str(feature_importance.max()) + ' min: ' + str(feature_importance.min())
print '---' * 10
