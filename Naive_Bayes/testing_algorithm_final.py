from naive_bayes import Bayes_Classifier
import numpy as np
import re
import os


def testing_algorithm(trainDirectory = "./db_txt_files/", validationDirectory = "./movie_reviews/"):
    #CONFUSION MATRIX
    confusionMatrix = np.zeros((3,3), dtype=int)

    #CALLING CLASSIFIER
    bayes_c = Bayes_Classifier(trainDirectory)

    #DATA PROCESSING FOR CLASSIFIER
    IFileList =[]
    for fFileObj in os.walk(validationDirectory):
        IFileList = fFileObj[2]

    #VARIABLES INTIALIZE
    accurateCount, count, neg_total, pos_total, neu_total = 0, 0, 0, 0, 0

    # MAIN TESTING ALGORITHM
    for i in range(len(IFileList)):
        count += 1
        rating_for_files = re.search('-', IFileList[i]).start()
        rating = int(IFileList[i][rating_for_files + 1])
        file = bayes_c.loadFile(validationDirectory + IFileList[i])
        nrci = [2,3,4] #Neutral rating compare index
        prediction = bayes_c.classify(file)

#       #FINDING RATINGS FOR REVIEWS
        if rating == prediction:
            accurateCount += 1
        elif (rating in nrci) and prediction == 3:
            accurateCount += 1
        
        #TOTAL RATINGS COUNTER
        if rating == 1:
            neg_total += 1
        elif rating == 5:
            pos_total += 1
        else:
            neu_total += 1

        # GENERATING CONFUSION MATRIX
        # FOR NEGATIVE
        if rating == 1 and prediction == 1:
            confusionMatrix[0,0] += 1
        elif rating == 1 and prediction == 3:
            confusionMatrix[0,1] += 1
        elif rating == 1 and prediction == 5:
            confusionMatrix[0,2] += 1
        # FOR NEUTRAL
        elif (rating in nrci) and prediction == 1:
            confusionMatrix[1,0] += 1
        elif (rating in nrci) and prediction == 3:
            confusionMatrix[1,1] += 1
        elif (rating in nrci) and prediction == 5:
            confusionMatrix[1,2] += 1
        # FOR POSITIVE
        elif rating == 5 and prediction == 1:
            confusionMatrix[2,0] += 1
        elif rating == 5 and prediction == 3:
            confusionMatrix[2,1] += 1
        elif rating == 5 and prediction == 5:
            confusionMatrix[2,2] += 1

    print("---------------------RESULT------------------------")

    #DISPLAY CONFUSION MATRIX AND RESULTS
    print("\nCONFUSION MATRIX")
    print(confusionMatrix)
    print(f"\nActual Negative Reviews: {neg_total}")
    print(f"Actual Neutral Reviews: {neu_total}")
    print(f"Actual Positive Reviews: {pos_total}")

    neg_Recall = confusionMatrix[0,0] / (confusionMatrix[0,0]+confusionMatrix[0,1]+confusionMatrix[0,2])
    neg_Precision = confusionMatrix[0,0] / (confusionMatrix[0,0]+confusionMatrix[1,0]+confusionMatrix[2,0])

    pos_Recall = confusionMatrix[2,2] / (confusionMatrix[2,0]+confusionMatrix[2,1]+confusionMatrix[2,2])
    pos_Precision = confusionMatrix[2,2] / (confusionMatrix[0,2]+confusionMatrix[1,2]+confusionMatrix[2,2])

    if neu_total == 0:
        Precision = (pos_Precision + neg_Precision) / 2
        Recall = (pos_Recall + neg_Recall) / 2
        Fmeasure = (2 * Precision * Recall) / (Precision + Recall)
    else:
        neutralRecall = confusionMatrix[1, 1] / (confusionMatrix[1, 0] + confusionMatrix[1, 1] + confusionMatrix[1, 2])
        neutralPrecision = confusionMatrix[1, 1] / (
                    confusionMatrix[0, 1] + confusionMatrix[1, 1] + confusionMatrix[2, 1])
        Precision = (pos_Precision + neutralPrecision + neg_Precision) / 3
        Recall = (pos_Recall + neutralRecall + neg_Recall) / 3
        Fmeasure = (2 * Precision * Recall) / (Precision + Recall)
        

    print(f"\nAccuracy: {(accurateCount/count)*100}")
    print(f"Precision: {Precision}")
    print(f"Recall: {Recall}")
    print(f"Fmeasure: {Fmeasure}")


if __name__ == "__main__":
    print("Validation Results:")
    testing_algorithm('db_txt_files/', 'movies_reviews/')
