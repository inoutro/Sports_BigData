import pandas as pd

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression

##### 공민서 대리님 데이터 셋 #####
# data_sam = pd.read_csv("sam_data_dummy_0912.csv", encoding = "cp949")

# X_col = ["race", "chulNo", "age", "wgBudam", "wgHr", "wgHrDiff", "rcCntY", "ord1CntY", "ord2CntY", "rank", "ord1PerT", "ord2PerT", "sex_female", "sex_gelding", "sex_male", "name_domestic", "name_foreign", "track_humidity_0", "track_humidity_1", "track_humidity_2", "track_humidity_3", "track_humidity_4"]

# Y_col = ["ord", "rcTime"]

##### 이성진 대리님 데이터 셋 #####
# data_sam = pd.read_csv("C:/Users/korea/Desktop/sam_data.csv", encoding = "cp949")

# X_col = ["race", "chulNo" , "age" ,	"wgBudam" ,	"ord1CntT" , "ord1CntY" ,	"ord2CntT" ,	"ord2CntY" ,	"rcCntT" ,	"rcCntY" ,	"chaksunT" ,	"chaksunY" ,	"chaksun_6m" ,	"rating" ,	"rcDist" ,	"trAge" ,	"ord1CntT_tr" ,	"ord2CntT_tr" ,	"rcCntT_tr" ,	"ord1CntY_tr" ,	"ord2CntY_tr" ,	"rcCntY_tr" ,	"age_jk" ,	"ord1CntT_jk" ,	"ord2CntT_jk" ,	"rcCntT_jk" ,	"ord1CntY_jk" ,	"ord2CntY_jk" ,	"rcCntY_jk" ,	"wg_Budam_jk" ,	"wg_Budam_others_jk" ,	"num_own" ,	"num_reg" ,	"num_cancel" ,	"chaksunT_ow" ,	"chaksunY_ow" ,	"ord1CntT_ow" ,	"ord2CntT_ow" ,	"ord3CntT_ow" ,	"rcCntT_ow" ,	"ord1CntY_ow" ,	"ord2CntY_ow" ,	"ord3CntY_ow" ,	"rcCntY_ow" ,	"from_recentRcDate" ,	'from_trDebut'	 , "from_jkDebut" ,	"from_owRegister"]

# Y_col = ["ord", "rcTime"]


##### 자체제작 데이터 셋 #####
data_sam = pd.read_csv("../input_data/자체 제작 데이터셋/sam_data_0901.csv")

X_col = ["race", "chulNo", "age", "wgBudam", "rank", "rcCntY", "ord1CntY", "ord2CntY", "chaksunT", "chaksunY",
         "chaksun_6m", "trAge", "rcCntY_tr", "ord1CntY_tr", "ord2CntY_tr", 'age_jk', "wg_Budam_jk", "rcCntY_jk",
         "ord1CntY_jk", "ord2CntY_jk", "num_own", "num_reg", "num_cancel", "chaksunT_ow", "chaksunY_ow", "ord1CntY_ow",
         "ord2CntY_ow", "ord3CntY_ow", "rcCntY_ow", "hr_ord1PerT", "hr_ord2PerT", "tr_ord1PerT", "tr_ord2PerT",
         "jk_ord1PerT", "jk_ord2PerT", "ow_ord1PerT", "ow_ord2PerT", "ow_ord3PerT", "tr_career", "jk_career",
         "ow_career", "sex_female", "sex_gelding", "sex_male", "name_domestic", "name_foreign"]

Y_col = ["ord", "rcTime"]

accuracy_list = []

# for acc in range(1,30):

randomforest_model = RandomForestClassifier(n_jobs=-1, random_state=0, n_estimators=1000)

kf = KFold(n_splits=5, shuffle=True)
for TR, TE in kf.split(np.unique(data_sam["race"])):
    TR, TE = TR + 1, TE + 1
    # print(TR)
    # print(len(TR))
    # print(TE)
    # print(len(TE))
    train = data_sam[data_sam["race"].isin(TR)]
    test = data_sam[data_sam["race"].isin(TE)]

    X_train = train[X_col]
    Y_train = train[Y_col]
    X_test = test[X_col]
    Y_test = test[Y_col]

    randomforest_model.fit(X_train[X_train.columns[1:]], Y_train["ord"])

    Y_pred_proba = randomforest_model.predict_proba(X_test[X_test.columns[1:]])
    Ques_proba = list(Y_pred_proba)

    Ques_proba_test = []
    for i in Ques_proba:
        Ques_proba_test.append(i[1])

    calculate_accuracy = pd.DataFrame(X_test, columns=["race"])
    calculate_accuracy["pred_ord_percent"] = Ques_proba
    calculate_accuracy["pred_ord_1_per"] = Ques_proba_test
    calculate_accuracy["pred_ord_1_per_minus"] = -calculate_accuracy["pred_ord_1_per"]
    calculate_accuracy["pred_ord_1_per_ranking"] = calculate_accuracy.groupby("race")["pred_ord_1_per_minus"].rank(
        method="first")
    calculate_accuracy["pred_ord_1_per_ranking"] = calculate_accuracy["pred_ord_1_per_ranking"].astype(int)
    calculate_accuracy["real_ord"] = Y_test["ord"]

    calculate_accuracy.loc[calculate_accuracy.pred_ord_1_per_ranking == 3, ["pred_ord_1_per_ranking"]] = 1
    calculate_accuracy.loc[calculate_accuracy.pred_ord_1_per_ranking == 2, ["pred_ord_1_per_ranking"]] = 1
    calculate_accuracy.loc[calculate_accuracy.pred_ord_1_per_ranking != 1, ["pred_ord_1_per_ranking"]] = 0

    # slicing data what i want
    modify_c_a = pd.DataFrame(calculate_accuracy, columns=["race", "real_ord", "pred_ord_1_per_ranking"])

    print(modify_c_a)

    # calculate accuracy
    predict_sum = 0
    urace = np.unique(modify_c_a.race)

    for i in urace:
        x = modify_c_a.loc[modify_c_a["race"] == i]
        y = modify_c_a.loc[modify_c_a["race"] == i]
        if all(x.real_ord == y.pred_ord_1_per_ranking):
            predict_sum = predict_sum + 1

    print(predict_sum)
    print(len(urace))

    accuracy = predict_sum / len(urace)
    print(accuracy)

    accuracy_list.append(accuracy)

print(accuracy_list)
print(sum(accuracy_list) / len(accuracy_list))
print(np.std(accuracy_list))
print(np.mean(accuracy_list))
