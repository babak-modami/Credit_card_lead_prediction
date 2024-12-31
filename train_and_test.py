# join both train and test data for preprocessing..
train['train'] = 1
test['train'] = 0

df = pd.concat([train,test],axis=0).reset_index(drop=True)

# check about the data type of each feature
df.info()

#check for missing values
df.isnull().sum()

df['Credit_Product'].value_counts(normalize = True)

# we have assigned a new category to the missing values of the credit_product because it might be that there association with bank might be of different nature?
df['Credit_Product'] = df['Credit_Product'].fillna('X')

le=LabelEncoder()
for col in ['Region_Code']:
    df[col]=  df[col].astype('str')
    df[col]= le.fit_transform(df[col])

# Introduced dummy fot the nominal categorical varibales
df= pd.concat([df,pd.get_dummies(df['Gender'],prefix = str('Gender'),prefix_sep='_')],axis = 1)
df = pd.concat([df,pd.get_dummies(df['Occupation'],prefix = str('Occupation'),prefix_sep='_')],axis = 1)
df = pd.concat([df,pd.get_dummies(df['Channel_Code'],prefix = str('Channel_Code'),prefix_sep='_')],axis = 1)
df = pd.concat([df,pd.get_dummies(df['Credit_Product'],prefix = str('Credit_Product'),prefix_sep='_')],axis = 1)
df = pd.concat([df,pd.get_dummies(df['Is_Active'],prefix = str('Is_Active'),prefix_sep='_')],axis = 1)

df.drop(['Gender'], axis = 1, inplace = True)
df.drop(['Occupation'], axis = 1, inplace = True)
df.drop(['Channel_Code'], axis = 1, inplace = True)
df.drop(['Credit_Product'], axis = 1 , inplace = True)
df.drop(['Is_Active'], axis = 1 , inplace = True)

df.head()

cat_features = ['Gender_Female','Gender_Male','Occupation_Entrepreneur','Occupation_Other','Occupation_Salaried','Occupation_Self_Employed','Channel_Code_X1','Channel_Code_X2','Channel_Code_X3','Channel_Code_X4','Credit_Product_No','Credit_Product_X','Credit_Product_Yes','Is_Active_No','Is_Active_Yes']


#I have used LightGBM model for training the data. I have used stratified Kfold(10) as cross validation technique.


def train_lgbm(df,seed,cat_features):
    X = df[df['train'] == 1]
    y = df[df['train'] == 1]['Is_Lead']
    test_data = df[df['train'] == 0]
    num_split = 10
    folds = StratifiedKFold(n_splits=num_split)
    excluded_features = ['Is_Lead','train','ID']
    train_features = [_f for _f in X.columns if _f not in excluded_features]
    importances = pd.DataFrame()
    oof_reg_preds = np.zeros( X.shape[0] )
    test_preds = np.zeros( test_data.shape[0] )
    for fold_, (trn_, val_) in enumerate( folds.split(X, y) ):
        print("Fold:",fold_)
        trn_x, trn_y = X[train_features].iloc[trn_], y.iloc[trn_]
        val_x, val_y = X[train_features].iloc[val_], y.iloc[val_]
        clf1 = LGBMClassifier(
            n_jobs = -1,
            learning_rate = 0.0094,
            n_estimators = 10000,
            colsample_bytree = 0.94,
            subsample = 0.75,
            subsample_freq = 1,
            reg_alpha= 1.0,
            reg_lambda = 5.0,
            random_state = seed
        )
        clf1.fit(
            trn_x,trn_y ,
            eval_set=[(val_x, val_y)],
            early_stopping_rounds = 100,
            verbose = False,
            eval_metric ='auc',
            categorical_feature = cat_features
        )
        
        # Extra boosting.
        clf = LGBMClassifier(
            n_jobs = -1,
            learning_rate = 0.00094,
            n_estimators = 10000,
            colsample_bytree = 0.94,
            subsample = 0.75,
            subsample_freq = 1,
            reg_alpha= 1.0,
            reg_lambda = 5.0,
            random_state = seed
        )
        clf.fit(
            trn_x,trn_y ,
            eval_set=[(val_x, val_y)],
            early_stopping_rounds=300,
            verbose=False,
            eval_metric='auc',
            categorical_feature = cat_features,
            init_model = clf1
        )
        
        imp_df = pd.DataFrame()
        imp_df['feature'] = train_features
        imp_df['importance'] = clf.booster_.feature_importance(importance_type='gain')

        imp_df['fold'] = fold_ + 1
        importances = pd.concat([importances, imp_df], axis=0, sort=False)

        oof_reg_preds[val_] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
        _preds = clf.predict_proba(test_data[train_features], num_iteration=clf.best_iteration_)[:, 1]
        test_preds += _preds
        print("fold"+str(fold_)+" auc",roc_auc_score(val_y, oof_reg_preds[val_]))
        del  trn_x, trn_y, val_x, val_y,trn_, val_
        gc.collect()



    test_preds = test_preds/num_split
    print(roc_auc_score(y, oof_reg_preds))
    
    return test_preds,importances,oof_reg_preds,clf,train_features

# train the model by passing data, random seed as 42 and categorical feature list
lgbm_preds,feat_imp_lgbm,oof_lgbm,lgbm_model,feats = train_lgbm(df,42,cat_features)

lgbm_preds1,feat_imp_lgbm1,oof_lgbm1,lgbm_model1,feats = train_lgbm(df,11,cat_features)

# Function to display feature importance...
def display_importances(feature_importance_df_,model):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:30].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(12, 8))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title(model+" Features (avg over folds)")
    plt.tight_layout()
    plt.savefig(model +"_importances-01.png")

display_importances(feat_imp_lgbm,"LGBM")

oof_ensemble = oof_lgbm*.50 + oof_lgbm1*.50
roc_auc_score(df[df['train']==1]['Is_Lead'],oof_ensemble)

final_preds = lgbm_preds*.50 + lgbm_preds1*.50
