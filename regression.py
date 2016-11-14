# -*- coding: utf-8 -*-

from pandas import read_csv, DataFrame, Series
from sklearn import cross_validation, svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from matplotlib import gridspec
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


def residuals_plot(y_true, y_pred, name):
    lr = LinearRegression()

    sorted_y_true, sorted_y_pred = zip(*sorted(zip(y_true, y_pred)))

    lr.fit(Series(y_true).to_frame(), y_pred)
    y_regression = lr.predict(Series(y_true).to_frame())

    # PLOT
    fig = plt.figure(figsize=(10, 5))
    fig.suptitle(name, fontsize=14, fontweight='bold')

    fig.text(0.07, 0.8,
              'R^2: %5.2f%%\nMSE: %5.2f' % (r2_score(y_true, y_pred) * 100, mean_absolute_error(y_true, y_pred)),
              style='italic',
              # transform = frame1.transAxes,
              bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10})

    # GRID 1
    gs1 = gridspec.GridSpec(4, 1)
    gs1.update(left=0.05, right=0.48, wspace=0.05)
    ax1 = plt.subplot(gs1[:-1, :])
    ax2 = plt.subplot(gs1[-1, :])

    # Plot Data-model
    ax1.plot(sorted_y_true, sorted_y_pred, 'ob', label="pred")
    ax1.plot(sorted_y_true, sorted_y_true, 'or', label="obj")
    ax1.plot(sorted_y_true, sorted_y_true, color='coral', linestyle='-', label="regressionOfObj")
    ax1.plot(y_true, y_regression, color='dodgerblue', linestyle = '-', label="regressionOfPred")
    ax1.legend(bbox_to_anchor=(1.05, 1.05)).get_frame().set_alpha(0.5)
    ax1.set_xticklabels([])
    ax1.set_ylabel('prediction value')
    ax1.grid()


    # Residual plot
    difference = map(lambda x, y: y - x, sorted_y_true, sorted_y_pred)
    ax2.plot(sorted_y_true, difference, 'og', label="residuals")
    ax2.legend(bbox_to_anchor=(1.05, 1.05)).get_frame().set_alpha(0.5)
    ax2.set_xlabel('obj value')
    ax2.grid()


    # resort
    # sorted_y_pred, sorted_y_true = zip(*sorted(zip(y_pred, y_true)))
    lr.fit(Series(y_pred).to_frame(), Series(y_true))
    y_regression = lr.predict(Series(y_pred).to_frame())

    #GRID2
    gs2 = gridspec.GridSpec(4, 1)
    gs2.update(left=0.55, right=0.98, wspace=0.05)
    ax3 = plt.subplot(gs2[:-1, :])
    ax4 = plt.subplot(gs2[-1, :])

    # Plot Data-model
    ax3.plot(sorted_y_pred, sorted_y_true, 'or', label="obj")
    ax3.plot(sorted_y_pred, sorted_y_pred, 'ob', label="pred")
    ax3.plot(sorted_y_pred, sorted_y_pred, color='dodgerblue', linestyle='-', label="regressionOfPred")
    ax3.plot(y_pred, y_regression, color='coral', linestyle='-', label="regressionOfObj")
    ax3.legend(bbox_to_anchor=(1.05, 1.05)).get_frame().set_alpha(0.5)
    ax3.set_xticklabels([])
    ax3.set_ylabel('obj value')
    ax3.grid()

    # Residual plot
    difference = map(lambda x, y: x - y, sorted_y_true, sorted_y_pred)
    ax4.plot(sorted_y_pred, difference, 'og', label="residuals")
    ax4.grid()
    ax4.set_xlabel('prediction value')
    ax4.legend(bbox_to_anchor=(1.05, 1.05)).get_frame().set_alpha(0.5)

    plt.savefig('plots/' + name + '.png')
    # plt.show()


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
    # residuals_plot(ROCtestTRG, pred, 'residuals-' + name)


feature_importance = models['RandomForestRegression'].feature_importances_
names_feature = list(train.columns.values)
f_i_zipped = zip(names_feature, feature_importance.tolist())
f_i_zipped.sort(key = lambda t:t[1], reverse=True)
print 'Влияние факторов [top 10], %:'
for n, f in f_i_zipped[:10]:
    print " %10f - %s" % (f, n)
print 'max: ' + str(feature_importance.max()) + ' min: ' + str(feature_importance.min())
print '---' * 10
