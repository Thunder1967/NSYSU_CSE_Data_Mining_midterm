import myKNN
import myUtil
import myPreprocess
import numpy as np
import made_by_AI

def testBestK_Kfold(X,Y,Kf,preprocess,distance_fnc):
    res = []
    fold = myUtil.Kfold(Kf,len(X))
    for train,test in fold:
        model = myKNN.BruteKNN(X[train],Y[train],preprocess,distance_fnc)
        X_test = model.preprocess.testPreprocess(X[test])
        tmp = []
        for k_val in range(3,100):
            model.setK(k_val)
            tmp.append([k_val,model.getTrainingAccuracy(),model.getAccuracy(X_test,Y[test])])
        res.append(tmp)
    return np.mean(np.array(res),axis=0)

if __name__=="__main__":
    preprocess = [
        (myPreprocess.Scale_IQRC_Preprocess(),"Scale_IQRC"),
        (myPreprocess.Scale_IQRR_Preprocess(),"Scale_IQRR"),
        (myPreprocess.Scale_Preprocess(),"Scale"),
        (myPreprocess.STD_IQRC_Preprocess(),"STD_IQRC"),
        (myPreprocess.STD_IQRR_Preprocess(),"STD_IQRR"),
        (myPreprocess.STD_Preprocess(),"STD"),
    ]
    distance_fnc = [
        (myUtil.euclidean_distance_sq,"euc"),
        (myUtil.manhattan_distance,"man")
    ]
    X,Y = myUtil.read_data("train.csv")
    # generateGraph(
    #             testBestK_Kfold(X,Y,5,preprocess[0][0],distance_fnc[0][0]),
    #             "test.png"
    #         )
    # exit()
    best_combination = []
    for i in preprocess:
        for j in distance_fnc:
            myUtil.timeTest()
            pictureName = f"knn_{i[1]}_{j[1]}.png"
            best_combination.append(made_by_AI.generateGraph(
                testBestK_Kfold(X,Y,10,i[0],j[0]),
                pictureName
            ))
            print(f"create {pictureName}")
            print(f'costs {myUtil.timeTest()} second\n')
    best_combination.sort(key=lambda x:x[2])
    for i in best_combination:
        print(i)