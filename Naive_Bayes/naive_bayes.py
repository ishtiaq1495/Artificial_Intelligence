import math, os, pickle, re


def list_capitilize(tokenized_list):
    new_list = [i.upper() for x in tokenized_list for i in x]
    return new_list


class Bayes_Classifier:

    def __init__(self, trainDirectory="db_txt_files/"):
        # INTIALIZING THE REQUIRED VARIABLES
        self.trainDirectory = trainDirectory
        self.neutral_dict, self.neg_dict, self.post_dict = {}, {}, {}
        self.positive_word_count, self.negative_word_count, self.neutral_word_count, self.positive_file_count, \
        self.negative_file_count, self.neutral_file_count, self.total_file_count = 0, 0, 0, 0, 0, 0, 0

        # CREATING LIST OF ALL DOCUMENTS IN TARGET DIRECTORY
        iFileList = []
        for fFileObj in os.walk(self.trainDirectory):
            iFileList = fFileObj[2]

        # DIVINDING DOCUMENTS IN POSTIVIE, NEGATIVE AND NEUTRAL LISTS
        self.negative_review = [iFileList[x] for x in range(len(iFileList)) if '-1-' in iFileList[x]]
        self.positive_review = [iFileList[x] for x in range(len(iFileList)) if '-5-' in iFileList[x]]
        self.neutral_review = [y for y in [x for x in iFileList if x not in self.negative_review] if
                               y not in self.positive_review]

        # COUNTING DOCUMENTS OF EACH SEGMENT
        self.positive_file_count = len(self.positive_review)
        self.negative_file_count = len(self.negative_review)
        self.neutral_file_count = len(self.neutral_review)

        # CHECKING IF DICTIONARY ALREADY PRESENT IF NOT THE CLASSIFIER IS TRAINED TO FORM TH DICTIONARY
        try:
            self.post_dict = self.load('dictionary/postive_reviews_dic')
            self.neg_dict = self.load('dictionary/negative_reviews_dic')
            self.neutral_dict = self.load('dictionary/neutral_reviews_dic')
        except OSError:
            self.train()

        # COUNTING THE WORDS IN EACH DICTIONARY
        for i in self.post_dict:
            self.positive_word_count += self.post_dict[i]
        for i in self.neg_dict:
            self.negative_word_count += self.neg_dict[i]
        for i in self.neutral_dict:
            self.neutral_word_count += self.neutral_dict[i]

        self.total_word_count = self.positive_word_count + self.negative_word_count + self.neutral_word_count
        self.total_file_count = len(iFileList)

        # CALCULATING THE PRIOR VALUES FOR EACH SEGMNT
        self.positivePrior = (self.positive_file_count / self.total_file_count)
        self.negativePrior = (self.negative_file_count / self.total_file_count)
        self.neutralPrior = (self.neutral_file_count / self.total_file_count)

    def train(self):
        # TOKENIZING THE TXT FILES IN EACH SGMENT AND FORMING A LIST OF WORDS
        tokenize_neg = [self.tokenize(self.loadFile(f'{self.trainDirectory}/{self.negative_review[x]}')) for x in
                        range(len(self.negative_review))]
        tokenize_pos = [self.tokenize(self.loadFile(f'{self.trainDirectory}/{self.positive_review[x]}')) for x in
                        range(len(self.positive_review))]
        tokenize_neutral = [self.tokenize(self.loadFile(f'{self.trainDirectory}/{self.neutral_review[x]}')) for x in
                            range(len(self.neutral_review))]

        # CAPITILZING THE LIST OF WORDS
        t_p = list_capitilize(tokenize_pos)
        t_n = list_capitilize(tokenize_neg)
        t_ne = list_capitilize(tokenize_neutral)

        # FORMING THE DICTIONARY FOR POSITIVEE, NEGATIVE AND NEUTRAL REVIEWS
        for i in t_p:
            self.post_dict[i] = self.post_dict.get(i, 0) + 1
        for i in t_n:
            self.neg_dict[i] = self.neg_dict.get(i, 0) + 1
        for i in t_ne:
            self.neutral_dict[i] = self.neutral_dict.get(i, 0) + 1

        # ADDING WORDS THAT ARE PRESENT ONE DICTIONARY BUT NOT IN THE OTHER
        for i in self.post_dict:
            if i not in self.neg_dict:
                self.neg_dict[i] = 1
            if i not in self.neutral_dict:
                self.neutral_dict[i] = 1

        for i in self.neutral_dict:
            if i not in self.post_dict:
                self.post_dict[i] = 1
            if i not in self.neg_dict:
                self.neg_dict[i] = 1

        for i in self.neg_dict:
            if i not in self.post_dict:
                self.post_dict[i] = 1
            if i not in self.neutral_dict:
                self.neutral_dict[i] = 1

        # SAVING THE DICTIONARY TO BE LOADED FOR LATER USE
        self.save(self.post_dict, 'dictionary/postive_reviews_dic')
        self.save(self.neg_dict, 'dictionary/negative_reviews_dic')
        self.save(self.neutral_dict, 'dictionary/neutral_reviews_dic')

    # FUCNTION TO CAPITILIZA THE WORDS IN THE LIST

    def classify(self, sText):
        '''Given a target string sText, this function returns the most likely document
      class to which the target string belongs. This function should return one of three
      strings: "positive", "negative" or "neutral".
      '''
        # TOKENIZING THE TEXT INPUTTED BY THE USER AND CAPITILIZING THEM
        tokenized_stext = self.tokenize(sText)
        cap_stext = [i.upper() for i in tokenized_stext]

        # FINDING THE LOG BASE 10 VALUES OF THE PRIOR VALUES
        prob_of_positive = math.log10(self.positivePrior)
        prob_of_negative = math.log10(self.negativePrior)
        prob_of_neutral = math.log10(self.neutralPrior)

        # CHECKING THE TOTAL NUMBER OF THE WORDS PRESENT FROM THE SENTENCE ENTERED, IF WORD IN SENTENCE NOT PRESENT ADD ONE
        for i in cap_stext:
            if i in self.post_dict:
                total_positive_word = self.post_dict[i]
            else:
                total_positive_word = 1
            if i in self.neg_dict:
                total_negative_word = self.neg_dict[i]
            else:
                total_negative_word = 1
            if i in self.neutral_dict:
                total_neutral_word = self.neutral_dict[i]
            else:
                total_neutral_word = 1

            # CALCULATING LIKELIHOOD FOR EACH TYPE
            likelihoodPositive = math.log10((total_positive_word + 1) / (self.positive_word_count + 1))
            likelihoodNegative = math.log10((total_negative_word + 1) / (self.negative_word_count + 1))
            likelihoodNeutral = math.log10((total_neutral_word + 1) / (self.neutral_word_count + 1))

            # CALCULATING THE PROBABILTIY OF EACH TYPE
            prob_of_positive += likelihoodPositive
            prob_of_negative += likelihoodNegative
            prob_of_neutral += likelihoodNeutral

        # FINDING DIFFERENCE BETWEEN POSITIVE AND NEGATIVE TO DECIDE IF REVIEW IS NEUTRAL VALUE
        difference_in_rating = abs(abs(prob_of_positive) - abs(prob_of_negative))

        # MAKING PREDICTION BASED ON PROBABILTIY
        if difference_in_rating < 0.30:
            return 3
        elif prob_of_positive > prob_of_negative:
            return 5
        else:
            return 1

    def loadFile(self, sFilename):
        f = open(sFilename, "r")
        sTxt = f.read()
        f.close()
        return sTxt

    def save(self, dObj, sFilename):
        f = open(sFilename, "wb")
        p = pickle.Pickler(f)
        p.dump(dObj)
        f.close()

    def load(self, sFilename):
        f = open(sFilename, "rb")
        u = pickle.Unpickler(f)
        dObj = u.load()
        f.close()
        return dObj

    def tokenize(self, sText):
        lTokens = []
        sToken = ""
        for c in sText:
            if re.match("[a-zA-Z0-9]", str(c)) is not None or c == "\'" or c == "_" or c == '-':
                sToken += c
            else:
                if sToken != "":
                    lTokens.append(sToken)
                    sToken = ""
                if c.strip() != "":
                    lTokens.append(str(c.strip()))

        if sToken != "":
            lTokens.append(sToken)
        return lTokens


if __name__ == "__main__":
    bc = Bayes_Classifier()
    result = bc.classify("This movie is great")
    print(result)
