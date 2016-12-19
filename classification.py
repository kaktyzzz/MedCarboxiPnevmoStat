# -*- coding: utf-8 -*-
#https://habrahabr.ru/post/202090/
#http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

from pandas import read_csv, DataFrame, Series
from sklearn import cross_validation, svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, accuracy_score, r2_score
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import LabelEncoder
from scipy import interp
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import math

def stat(true, pred, classificator_name, class_names):

    print 'Classificator report for || ' + classificator_name
    print classification_report(true, pred, target_names=class_names)
    confusion = confusion_matrix(true, pred)
    for i in range(0, len(confusion)):
        tp = confusion[i][i]
        tn = 0
        fp = 0
        fn = 0
        for j in range(0, len(confusion)):
            tn += confusion[j][j]
            fn += confusion[i][j]
            fp += confusion[j][i]
        tn -= tp
        fp -= tp
        fn -= tp

        se = tp * 1.0 / (tp + fn)  # = tpr
        sp = tn * 1.0 / (tn + fp)  # = tnr
        ac = (tp + tn) * 1.0 / (tp + tn + fp + fn)
        pvp = tp * 1.0 / (tp + fp)  # precision
        pvn = tn * 1.0 / (tn + fn)

        print '%31s: se = %5f sp = %5f ac = %5f +pv = %5f -pv = %5f' % (class_names[i], se, sp, ac, pvp, pvn)

    print 'Точность предсказания, среднее: %5f' % accuracy_score(true, pred)
    print 'R^2 score: %5f' % r2_score(true, pred)
    print 'Истинные значения     ',true
    print 'Предсказанные значения', pred
    print ''

def ROCanalize(classificator_name, test, prob, pred):
    fpr = dict()
    tpr = dict()
    trhd = dict()
    roc_auc = dict()
    class_names = ['class 1: 0', 'class 2: -', 'class 3: +', 'class 4: +!']
    # class_names = ['class 1: 0', 'class 2', 'class 3:' 'class 4: -', 'class 5: +', 'class 6: +!']

    pl.figure()

    stat(test, pred, classificator_name, class_names)

    test_bin = label_binarize(test, classes=[1, 2, 3, 4, 5, 6])
    n_classes = test_bin.shape[1]

    # for i in range(n_classes):
    #     if i in test: # ЕСли есть такой классс в трушной выборке
    #         fpr[i], tpr[i], trhd[i] = roc_curve(test_bin[:, i], prob[:, i])
    #         roc_auc[i] = auc(fpr[i], tpr[i])
    #         pl.plot(fpr[i], tpr[i], label='%s (area = %0.2f)' % (class_names[i], roc_auc[i]))

    # fpr["micro"], tpr["micro"], trhd["micro"] = roc_curve(test_bin.ravel(), prob.ravel())
    # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # pl.plot(fpr["micro"], tpr["micro"],
    #         label='micro-average ROC curve (area = {0:0.2f})'
    #               ''.format(roc_auc["micro"]),
    #         linewidth=2)
    #
    # # Compute macro-average ROC curve and ROC area
    #
    # # First aggregate all false positive rates
    # all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    #
    # # Then interpolate all ROC curves at this points
    # mean_tpr = np.zeros_like(all_fpr)
    # for i in range(n_classes):
    #     mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    #
    # # Finally average it and compute AUC
    # mean_tpr /= n_classes
    # fpr["macro"] = all_fpr
    # tpr["macro"] = mean_tpr
    # roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # pl.plot(fpr["macro"], tpr["macro"],
    #          label='macro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["macro"]),
    #          linewidth=2)

    # pl.plot([0, 1], [0, 1], 'k--')
    # pl.xlim([0.0, 1.0])
    # pl.ylim([0.0, 1.0])
    # pl.xlabel('False Positive Rate')
    # pl.ylabel('True Positive Rate')
    # pl.title('ROC ' + classificator_name)
    # pl.legend(loc=0, fontsize='small')
    # pl.savefig('ROC11/' + classificator_name + '.png')
    # pl.show()



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

label.fit(data["vremya do gospitalizacii (chasi)"].drop_duplicates())
encode_dict["vremya do gospitalizacii (chasi)'"] = list(label.classes_)
data["vremya do gospitalizacii (chasi)"] = label.transform(data["vremya do gospitalizacii (chasi)"])

data.to_csv('DATA_encode.csv', sep=';')
for k, v in encode_dict.iteritems():
    print k + ":[" + ", ".join(v) + "]"

# data.pivot_table('№', ['Operirujushhij hirurg'], "Ajerobnost'", 'count').fillna(0).plot(kind='bar', stacked=True).get_figure().savefig('plots/hirurg_aerob.png')
# data.pivot_table('№', ['Vremja operacii'], "Ajerobnost'", 'count').fillna(0).plot().get_figure().savefig('plots/vremya_aerob.png')
# data.pivot_table('№', ['Vozrast'], "Ajerobnost'", 'count').fillna(0).plot(kind='bar', stacked=True).get_figure().savefig('plots/vosrast_aerob.png')
# data.pivot_table('№', ['Mesjac operacii'], "Ajerobnost'", 'count').fillna(0).plot(kind='bar', stacked=True).get_figure().savefig('plots/mesyac_aerob.png')

data.pivot_table('№', ['Operirujushhij hirurg'], "Rezul'tat1", 'count').fillna(0).plot(kind='bar', stacked=True).get_figure().savefig('plots/hirurg_res1.png')
data.pivot_table('№', ['Vremja operacii'], "Rezul'tat1", 'count').fillna(0).plot().get_figure().savefig('plots/vremya_res1.png')
data.pivot_table('№', ['Vozrast'], "Rezul'tat1", 'count').fillna(0).plot(kind='bar', stacked=True).get_figure().savefig('plots/vosrast_res1.png')
data.pivot_table('№', ['Mesjac operacii'], "Rezul'tat1", 'count').fillna(0).plot(kind='bar', stacked=True).get_figure().savefig('plots/mesyac_res1.png')
data.pivot_table('№', ["Vysejannaja kul'tura"], "Rezul'tat1", 'count').fillna(0).plot(kind='bar', stacked=True).get_figure().savefig('plots/vysev_res1.png')
data.pivot_table('№', ['Vremya prebivaniya v stacionare (dni)'], "Rezul'tat1", 'count').fillna(0).plot(kind='bar', stacked=True).get_figure().savefig('plots/stacionar_res1.png')

# data.pivot_table('PassengerId', ['Parch'], 'Survived', 'count').plot(ax=axes[1], title='Parch')
plt.show()

data = data.drop(['№', "Bol'noj", "Nomer istorii bolezni", "Data operacii", "Chuvstvitel'nost' k ab", "Ustojchivost' k a/b"], axis=1)

train = data.drop(["Rezul'tat", "Ajerobnost'", "Rezul'tat1"], axis=1) #DROP TARGET
target = data["Rezul'tat1"]


kfold = 10 #количество подвыборок для валидации
itog_val = {} #список для записи результатов кросс валидации разных алгоритмов
models = {}
getlist = []
name_groups = ''

# train = data.iloc[:, 2:]
models['RandomForestClassifier'] = RandomForestClassifier(n_estimators=1000) #в параметре передаем кол-во деревьев
models['ExtraTreesClassifier'] = ExtraTreesClassifier(n_estimators=1000)
models['KNeighborsClassifier'] = KNeighborsClassifier(n_neighbors=20) #в параметре передаем кол-во соседей
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
    probas = fit.predict_proba(ROCtestTRN)
    pred = fit.predict(ROCtestTRN)
    ROCanalize(name_groups + ' | ' + name, ROCtestTRG, probas, pred)

feature_importance = models['RandomForestClassifier'].feature_importances_
names_feature = list(train.columns.values)
f_i_zipped = zip(names_feature, feature_importance.tolist())
f_i_zipped.sort(key = lambda t:t[1], reverse=True)
print 'Влияние факторов [top 10], %:'
for n, f in f_i_zipped[:10]:
    print " %10f - %s" % (f, n)
print 'max: ' + str(feature_importance.max()) + ' min: ' + str(feature_importance.min())
print '---' * 10
print ''
print ''
