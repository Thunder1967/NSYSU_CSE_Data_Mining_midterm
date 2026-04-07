import myKNN
import myUtil
import myPreprocess
import made_by_AI

def testBestK(KNN1, X_test,Y_test):
    res = []
    X_test = KNN1.preprocess.testPreprocess(X_test)
    for i in range(3,100):
        KNN1.setK(i)
        res.append((i,KNN1.getTrainingAccuracy(),KNN1.getAccuracy(X_test,Y_test)))
    return res

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
    X_train,Y_train = myUtil.read_data_remove_Dimension("train.csv")
    X_test,Y_test = myUtil.read_data_remove_Dimension("test.csv")
    made_by_AI.generateGraph(
                testBestK(
                    myKNN.BruteKNN(
                        X_train,Y_train,
                        myPreprocess.STD_IQRC_Preprocess(),
                        myUtil.euclidean_distance_sq,0),
                    X_test,Y_test),
                    "test.png"
            )
    exit()
    for i in preprocess:
        for j in distance_fnc:
            pictureName = f"knn_{i[1]}_{j[1]}.png"
            made_by_AI.generateGraph(
                testBestK(
                    myKNN.BruteKNN(X_train,Y_train,i[0],j[0],0),
                    X_test,Y_test),
                    pictureName
            )
            print(f"create {pictureName}")