import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import cross_val_score
import pickle
import json

class BioModel:
    def __init__(self, k, classifier_name, negative_type='auto', seed=0):

        assert classifier_name in ['rf', 'xgb', 'cb'], 'Invalid classifier name'
        assert negative_type in ['auto', 'random'], 'Invalid negative type'

        self.k = k
        self.classifier_name = classifier_name
        self.classifier = None
        self.best_params = None
        self.df_positives = pd.read_csv(f'./datasets/processed/k{self.k}/df_positives_kmer.csv')
        self.df_negatives = pd.read_csv(f'./datasets/processed/k{self.k}/df_negatives_kmer.csv') if negative_type == 'auto' else pd.read_csv(f'./datasets/processed/k{self.k}/df_negatives_random_kmer.csv')
        self.seed = seed
        self.negative_type = negative_type

    def train_test_split(self):
        # Take the last 400 rows of the positives and negatives as test set
        df_positives_train = self.df_positives.iloc[:-400]
        df_positives_test = self.df_positives.iloc[-400:]
        df_negatives_train = self.df_negatives.iloc[:-400]
        df_negatives_test = self.df_negatives.iloc[-400:]

        self.X_train = pd.concat([df_positives_train, df_negatives_train])
        self.X_train = self.X_train.sample(frac=1).reset_index(drop=True)
        self.y_train = self.X_train['curation_status']
        self.y_train = self.y_train.replace({'positive': 1, 'negative': 0})
        self.X_train = self.X_train.drop(columns=['curation_status'])

        self.X_test = pd.concat([df_positives_test, df_negatives_test])
        self.X_test = self.X_test.sample(frac=1).reset_index(drop=True)
        self.y_test = self.X_test['curation_status']
        self.y_test = self.y_test.replace({'positive': 1, 'negative': 0})
        self.X_test = self.X_test.drop(columns=['curation_status'])

    def objective_rf(self, trial):
        n_estimators = trial.suggest_int('n_estimators', 2, 1000)
        max_depth = trial.suggest_int('max_depth', 1, 10)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        classifier_obj = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=0)
        scores = cross_val_score(classifier_obj, self.X_train, self.y_train, cv=10, scoring='accuracy')
        return scores.mean()
    
    def objective_xgb(self, trial):
        n_estimators = trial.suggest_int('n_estimators', 2, 1000)
        max_depth = trial.suggest_int('max_depth', 1, 10)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.5)
        gamma = trial.suggest_float('gamma', 0.01, 1.0)
        classifier_obj = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, gamma=gamma, random_state=0)
        scores = cross_val_score(classifier_obj, self.X_train, self.y_train, cv=10, scoring='accuracy')
        return scores.mean()
    
    def objective_cb(self, trial):
        n_estimators = trial.suggest_int('n_estimators', 2, 1000)
        max_depth = trial.suggest_int('max_depth', 1, 10)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.5)
        classifier_obj = cb.CatBoostClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=0, verbose=False)
        scores = cross_val_score(classifier_obj, self.X_train, self.y_train, cv=10, scoring='accuracy')
        return scores.mean()
    
    def optimize(self, n_trials=30):
        if self.classifier_name == 'rf':
            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.seed))
            study.optimize(self.objective_rf, n_trials=n_trials)
            self.best_params = study.best_params
            self.classifier = RandomForestClassifier(**self.best_params, random_state=0)
        elif self.classifier_name == 'xgb':
            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.seed))
            study.optimize(self.objective_xgb, n_trials=n_trials)
            self.best_params = study.best_params
            self.classifier = xgb.XGBClassifier(**self.best_params, random_state=0)
        elif self.classifier_name == 'cb':
            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.seed))
            study.optimize(self.objective_cb, n_trials=n_trials)
            self.best_params = study.best_params
            self.classifier = cb.CatBoostClassifier(**self.best_params, random_state=0, verbose=False)

    def train(self):
        self.classifier.fit(self.X_train, self.y_train)

    def save_model(self):
        with open(f'./results/k{self.k}/model{self.seed}_{self.classifier_name}_{self.negative_type}.pkl', 'wb') as f:
            pickle.dump(self.classifier, f)

    def save_best_params(self):
        with open(f'./results/k{self.k}/best_params{self.seed}_{self.classifier_name}_{self.negative_type}.json', 'w') as f:
            json.dump(self.best_params, f, indent=4)

    def evaluate(self, save_results=True):
        results = {}
        y_pred = self.classifier.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred)
        confusion = confusion_matrix(self.y_test, y_pred)
        results['accuracy'] = accuracy
        results['precision'] = precision
        results['recall'] = recall
        results['f1'] = f1
        results['roc_auc'] = roc_auc
        results['TP'] = int(confusion[1][1])
        results['TN'] = int(confusion[0][0])
        results['FP'] = int(confusion[0][1])
        results['FN'] = int(confusion[1][0])
        if save_results:
            with open(f'./results/k{self.k}/results{self.seed}_{self.classifier_name}_{self.negative_type}.json', 'w') as f:
                json.dump(results, f, indent=4)
        return results