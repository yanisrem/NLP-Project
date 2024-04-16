from src.preprocessing import pipeline_remove
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import class_weight
import numpy as np
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb

class RegexClassifier:
    def __init__(self, df, col_link) -> None:
        """
        Initialize the RegexClassifier object.

        Args:
            df (pandas.DataFrame): Input DataFrame.
            col_link (str): Name of the column containing links.
        """
        self.df = df
        self.col_link = col_link
    
    def predict(self):
        """
        Predict the presence of specific patterns in the link column.

        Returns:
            Series: Predicted values (1 for presence, 0 for absence).
        """
        condition_cdm = r"^chefmenageetveuve$|^perechef$|^cheff|^cheff$|^merechef$|^chefmen$|^chefmen|^cheffam$|^cheffam|^chm$|^chefmenage$|^chefmenage|^chmenage$|^chmenage|^chefm$|^chefm|^chmge$|^chmge|^chef$|^chef|^chefme$|^chefme|^cheffle$|^cheffle|^chme$|^chme|^chf$|^chf|^chefmaison$|^chefmaison|^chefmge$|^chefmge|^cheffamille$|^cheffamille|^chefveuf$|^chefveuf"
        df_pred_link_clean = self.df.copy()
        df_pred_link_clean[self.col_link] = df_pred_link_clean[self.col_link].apply(pipeline_remove)
        y_pred = df_pred_link_clean[self.col_link].str.contains(condition_cdm, regex=True).astype(int)
        return y_pred

class StoClassifier:
    def __init__(self, classifier, model_embedding = None, name_model_embedding = None, tokenizer = None):
        """
        Initialize the StoClassifier object.

        Args:
            classifier: The classifier model.
            model_embedding (optional): The embedding model.
            name_model_embedding (optional): Name of the embedding model: 'word2vec' or 'glove' or 'camemBERT'
            tokenizer (optional): Tokenizer for text data.
        """
        self.classifier = classifier
        self.model_embedding = model_embedding
        self.name_model_embedding = name_model_embedding
        self.tokenizer = tokenizer
        self.X_train = None
        self.X_test = None

    def fit_embedding(self, embedding_data, col_target, selectfeatures = False, return_x_train = True, return_x_test = True):
        """
        Fit the classifier with embedding data.

        Args:
            embedding_data: Instance of embedding data.
            col_target (str): Target column name.
            select_features (bool, default=False): Whether to select features.
            return_x_train (bool, default=True): Whether to return X_train.
            return_x_test (bool, default=True): Whether to return X_test.

        Returns:
            tuple or ndarray: X_train and X_test if specified.
        """
        X_train = embedding_data.embedding_train_dataframe(self.name_model_embedding, self.model_embedding, self.tokenizer)
        y_train = embedding_data.df_train_clean[col_target].values
        X_test = embedding_data.embedding_test_dataframe(self.name_model_embedding, self.model_embedding, self.tokenizer)
        y_test = embedding_data.df_test_clean[col_target].values

        if selectfeatures:
            selector = SelectFromModel(estimator = xgb.XGBClassifier(objective='binary:logistic'))
            X_train_new = selector.fit_transform(X_train, y_train)
            X_test_new = selector.transform(X_test)
            self.X_train = X_train_new
            self.X_test = X_test_new

        else:
            self.X_train = X_train
            self.X_test = X_test
        
        self.classifier.fit(self.X_train, y_train)

        if return_x_train and return_x_test:
            return self.X_train, self.X_test
        elif return_x_train and not return_x_test:
            return self.X_train
        elif return_x_test and not return_x_train:
            return self.X_test
    
    def fit_no_embedding(self, embedding_data, col_target, return_x_train = True, return_x_test = True):
        """
        Fit the classifier without embedding data.

        Args:
            embedding_data: Instance of embedding data.
            col_target (str): Target column name.
            return_x_train (bool, default=True): Whether to return X_train.
            return_x_test (bool, default=True): Whether to return X_test.

        Returns:
            tuple or ndarray: X_train and X_test if specified.
        """
        X_train = embedding_data.no_embedding_train_dataframe()
        y_train = embedding_data.df_train_clean[col_target].values
        X_test = embedding_data.no_embedding_test_dataframe()
        y_test = embedding_data.df_test_clean[col_target].values

        self.X_train = X_train
        self.X_test = X_test

        self.classifier.fit(self.X_train, y_train)

        if return_x_train and return_x_test:
            return self.X_train, self.X_test
        elif return_x_train and not return_x_test:
            return self.X_train
        elif return_x_test and not return_x_train:
            return self.X_test
    
    def predict(self, predict_proba = False):
        """
        Predict with the classifier.

        Args:
            predict_proba (bool, default=False): Whether to return probabilities.

        Returns:
            tuple or ndarray: Predictions.
        """
        if predict_proba :
            return self.classifier.predict(self.X_train), self.classifier.predict_proba(self.X_train)[:,1], self.classifier.predict(self.X_test), self.classifier.predict_proba(self.X_test)[:,1]

        else:
            return self.classifier.predict(self.X_train), self.classifier.predict(self.X_test)

class CustomXGBPredictor(BaseEstimator, TransformerMixin):
    def __init__(self,
                 sample_weight = False,
                 custom_threshold = 0.5,
                 learning_rate = 0.1,
                 n_estimators = 100, #Xgboost parameters
                 max_depth = 6,
                 max_leaves = 0,
                 subsample = 1,
                 colsample_bytree = 1,
                 gamma = 0,
                 min_child_weight = 1,
                 reg_lambda = 1):
        """
        Initialize the CustomXGBPredictor object.

        Args:
            sample_weight (bool, default=False): Whether to use sample weights.
            custom_threshold (float, default=0.5): Custom threshold for prediction.
            learning_rate (float, default=0.1): Learning rate for XGBoost.
            n_estimators (int, default=100): Number of estimators for XGBoost.
            max_depth (int, default=6): Maximum tree depth for XGBoost.
            max_leaves (int, default=0): Maximum number of leaves for XGBoost.
            subsample (float, default=1): Subsample ratio of the training instance for XGBoost.
            colsample_bytree (float, default=1): Subsample ratio of columns when constructing each tree for XGBoost.
            gamma (float, default=0): Minimum loss reduction required to make a further partition on a leaf node of the tree for XGBoost.
            min_child_weight (float, default=1): Minimum sum of instance weight (hessian) needed in a child for XGBoost.
            reg_lambda (float, default=1): L2 regularization term on weights for XGBoost.
        """
        
        self.sample_weight = sample_weight
        self.custom_threshold = custom_threshold
        #Xgboost parameters
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.model = xgb.XGBClassifier(learning_rate = self.learning_rate,
                                       n_estimators = self.n_estimators,
                                       max_depth = self.max_depth,
                                       max_leaves = self.max_leaves,
                                       subsample = self.subsample,
                                       colsample_bytree = self.colsample_bytree,
                                       gamma = self.gamma,
                                       min_child_weight = self.min_child_weight,
                                       reg_lambda = self.reg_lambda,
                                       objective = 'binary:logistic',
                                       random_state = 123)
    
    def fit(self, X, y = None):
        """
        Fit the CustomXGBPredictor model.

        Args:
            X (array-like): Input data.
            y (array-like, default=None): Target values.

        Returns:
            self: Fitted estimator.
        """
        if self.sample_weight:
           sample_weight = class_weight.compute_sample_weight(class_weight = 'balanced', y = y)
        else:
            sample_weight = None
        self.model.fit(X,y, sample_weight = sample_weight)


    def predict(self, X):
        """
        Predict the target values.

        Args:
            X (array-like): Input data.

        Returns:
            numpy.darray: Predicted class labels.
        """
        proba_pred = self.model.predict_proba(X)[:, 1]
        y_pred = (proba_pred >= self.custom_threshold).astype(int)
        return y_pred
    
    def predict_proba(self, X):
        """
        Predict the probability estimates.

        Args:
            X (array-like): Input data.

        Returns:
            numpy.darray: Predicted probability estimates.
        """
        return self.model.predict_proba(X)