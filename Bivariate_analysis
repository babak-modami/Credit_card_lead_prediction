#BIVARIATE ANALYSIS: CONTINUOUS CATEGORICAL VARIABLES

#List of Hypothesis and investigation to perform under this combination.  
#1) Do age plays a significant role in deciding whether one will take creait card service or not?
#2) Do duration of association of customers matter in activating credit card service?
#3) Can avg account balance play a role in deciding whether customer will take credit card or not?
  
def TwoSampZ(X1, X2, sigma1, sigma2, N1, N2):
    '''
  takes mean, standard deviation, and number of observations and returns p-value calculated for 2-sampled Z-Test
  '''
    from numpy import sqrt, abs, round
    from scipy.stats import norm
    ovr_sigma = sqrt(sigma1**2/N1 + sigma2**2/N2)
    z = (X1 - X2)/ovr_sigma
    pval = 2*(1 - norm.cdf(abs(z)))
    return pval
def TwoSampT(X1, X2, sd1, sd2, n1, n2):
    
    from numpy import sqrt, abs, round
    from scipy.stats import t as t_dist
    ovr_sd = sqrt(sd1**2/n1 + sd2**2/n2)
    t = (X1 - X2)/ovr_sd
    df = n1+n2-2
    pval = 2*(1 - t_dist.cdf(abs(t),df))
    return pval

def Bivariate_cont_cat(train, cont, cat, category):
    
  #creating 2 samples
    x1 = train[cont][train[cat]==category][:]
    x2 = train[cont][~(train[cat]==category)][:]
  
  #calculating descriptives
    n1, n2 = x1.shape[0], x2.shape[0]
    m1, m2 = x1.mean(), x2.mean()
    std1, std2 = x1.std(), x2.mean()
  
  #calculating p-values
    t_p_val = TwoSampT(m1, m2, std1, std2, n1, n2)
    z_p_val = TwoSampZ(m1, m2, std1, std2, n1, n2)

  #table
    table = pd.pivot_table(data=train, values=cont, columns=cat, aggfunc = np.mean)

  #plotting
    plt.figure(figsize = (15,6), dpi=140)
  
  #barplot
    plt.subplot(1,2,1)
    sns.barplot([str(category),'not {}'.format(category)], [m1, m2])
    plt.ylabel('mean {}'.format(cont))
    plt.xlabel(cat)
    plt.title('t-test p-value = {} \n z-test p-value = {}\n {}'.format(t_p_val,
                                                                z_p_val,
                                                                table))

  # boxplot
    plt.subplot(1,2,2)
    sns.boxplot(x=cat, y=cont, data=train)
    plt.title('categorical boxplot')

Bivariate_cont_cat(train, 'Age','Is_Lead', 1)

Bivariate_cont_cat(train, 'Vintage','Is_Lead', 1)

Bivariate_cont_cat(train, 'Avg_Account_Balance','Is_Lead', 1)

##List of all the hypothesis:-
#1 4) Do credit card issude or not depends on gender?
#2) Is credit of any type leads to take credit card?
#3) Do occupation of individuals plays an important role in deciding whether one will take credit card or not?
#4) Do active customers have high frequency of taking credit card service?
#5) Are acqistion channel an important source in deciding whether customer will take service or not?
#6) Is region of customer an important factor in deciding the activation of service?
#Bivariate : Categorical-Categorical
def BVA_categorical_plot(data, tar, cat):
    '''take data and two categorical variables,
  calculates the chi2 significance between the two variables 
  and prints the result with countplot & CrossTab
  '''
  #isolating the variables
    data = train[[cat,tar]][:]

  #forming a crosstab
    table = pd.crosstab(train[tar],train[cat],)
    f_obs = np.array([table.iloc[0][:].values,
                    table.iloc[1][:].values])

  #performing chi2 test
    from scipy.stats import chi2_contingency
    chi, p, dof, expected = chi2_contingency(f_obs)
  
  #checking whether results are significant
    if p<0.05:
        sig = True
    else:
        sig = False

  #plotting grouped plot
    sns.countplot(x=cat, hue=tar, data= train)
    plt.title("p-value = {}\n difference significant? = {}\n".format(round(p,8),sig))

  #plotting percent stacked bar plot
  #sns.catplot(ax, kind='stacked')
    ax1 = train.groupby(cat)[tar].value_counts(normalize=True).unstack()
    ax1.plot(kind='bar', stacked='True',title=str(ax1))
    int_level = train[cat].value_counts()

BVA_categorical_plot(train, 'Gender', 'Is_Lead')

BVA_categorical_plot(train, 'Region_Code', 'Is_Lead')

BVA_categorical_plot(train, 'Credit_Product', 'Is_Lead')

BVA_categorical_plot(train, 'Is_Active', 'Is_Lead')

BVA_categorical_plot(train, 'Channel_Code', 'Is_Lead')

BVA_categorical_plot(train, 'Occupation', 'Is_Lead')
