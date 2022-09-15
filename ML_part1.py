import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import numpy as np



df = pd.read_csv("Xy_train.csv")

print(df.isna().sum(axis = 0)) #gives the amount of NA values for each variable


def print_invalid_values(df):
    for i in range(len(df)):
        if df.iloc[i, 1] > 115 or df.iloc[i, 1] < 0:
            print("Invalid Age, Index:", i+2)
            #df.drop(i)
        if df.iloc[i, 2] not in (0,1):
            print("Invalid gender, Index:", i+2)
        if df.iloc[i, 3] not in (0,1,2,3):
            print("Invalid cp value, Index:", i + 2)
        if df.iloc[i, 4] > 200 or df.iloc[i, 4] < 80:
            print("Invalid blood pressure, Index:", i+2)
        if df.iloc[i, 5] > 360:     #serum cholesterol - HDL + LDL + 20% triglyceride level
            print("Invalid chol value, Index:", i + 2)    #Triglycerides 500 very high, LDL - 200 very high, HDL - more = better
            # probably HDL less than 40 for people with high LDL and trig
        if df.iloc[i, 6] not in (0,1):
            print("Invalid fasting blood sugar, Index:", i + 2)  # 0 means normal 1 means possible problam
        if df.iloc[i, 7] not in (0,1,2):
            print("Invalid resting electrocardiographic results, Index:", i + 2)  # 0 means normal 1 means possible problem
        if df.iloc[i, 8] > 220-df.iloc[i, 1]+20 or df.iloc[i, 8] < 80:
            print("Invalid heart rate value, Index:", i+2, df.iloc[i, 8] - (220-df.iloc[i, 1]),"higher than expected")
        if df.iloc[i, 9] not in (0,1):
            print("Invalid exercise angina value, Index:", i+2)
        if df.iloc[i, 10] < 0:
            print("Invalid ST depression, Index:", i+2)
        if df.iloc[i, 11] not in (0,1,2):
            print("Invalid slope, Index:", i + 2)
        if df.iloc[i, 12] not in (0,1,2,3,4):
            print("Invalid major vessels value, Index:", i + 2)
        if df.iloc[i, 13] not in (0,1,2,3):
            print("Invalid thalium value, Index:", i + 2)
    return df


df_copy = df[df['age']<=115]

#print_invalid_values(df)
print_invalid_values(df_copy)

corrMatrix = df.iloc[0:,1:].corr()

mask = np.zeros_like(corrMatrix, dtype=np.bool)
mask[np.triu_indices_from(mask)]= True
'''
f, ax = plt.subplots(figsize=(11, 15))
heatmap = sn.heatmap(corrMatrix,
                      mask = mask,
                      square = True,
                      linewidths=.5,
                      cmap= 'coolwarm',
                      cbar_kws = {'shrink': .4,
                      'ticks': [-1, -.5, 0, 0.5, 1]},
                      vmin = -1,
                      vmax = 1,
                      annot = True,
                      annot_kws = {"size": 12})

ax.set_yticklabels(corrMatrix.columns, rotation = 0)
ax.set_xticklabels(corrMatrix.columns)
heatmap.get_figure().savefig('heatmap.png', bbox_inches='tight')
'''



corrMatrix2 = df_copy.iloc[0:,1:].corr()
corr_values = corrMatrix[abs(corrMatrix)>=0.5]
print(corr_values.notna().sum()-1) #gives the amount of variables corelated |corr|>0.5 with each variable

sn.heatmap(corrMatrix2,  mask = mask, annot=True,cmap= 'coolwarm', vmin = -0.6, vmax = 0.6)
plt.show()      #corr matrix

low_corr = corrMatrix[abs(corrMatrix)<=0.1]
low_corr_y = low_corr['y']
print(low_corr_y[low_corr_y.notna()])


'''

'''
bins_list = [0, 40, 60, 80, 2000] #dismiss observations with age > 115
plt.hist(df['age'], bins = bins_list, density = True, stacked = True, facecolor = "b", edgecolor="#6A9662")
#plt.hist((28,60,12),(15,64,115))
plt.title('Age Histogram')
plt.xlabel("Age")
plt.ylabel("Probability")
#plt.legend(loc='upper right')
plt.show()

plt.hist(df['age'], bins = 'auto', facecolor = "b")
plt.title('Age Histogram')
plt.xlabel("Probability")
plt.show()



gender_count = df['gender'].value_counts()
plt.bar(('Female','Men'),(gender_count[0],gender_count[1]), width=0.5)
plt.title('Men and Female Distribution')
plt.show()

cp_count = df['cp'].value_counts()
ratio_cp = np.around(cp_count/len(df), decimals = 3)
cp_types = ['typical angina','atypical angina','non-anginal pain','asymptomatic']
plt.bar(cp_types,(ratio_cp[0],ratio_cp[1],ratio_cp[2],ratio_cp[3]), width=0.5)
plt.title('Chest pain types probability')
plt.show()


plt.hist(df['trestbps'], bins = 'auto', density = True, stacked = True, facecolor = "b", edgecolor="#6A9662")
plt.title('Blood Pressure Histogram')
plt.xlabel("resting blood pressure (mm/Hg)")
plt.xticks(ticks=(120,130,140,180), labels=("Normal","Elevated","High blood pressure","Hypertensive crisis"))
plt.axvline(x=120, ymin=0, ymax=1, color='g')
plt.axvline(x=130, ymin=0, ymax=1, color='y')
plt.axvline(x=140, ymin=0, ymax=1, color='r')
plt.axvline(x=180, ymin=0, ymax=1, color='#5c0e14', linewidth=4)
plt.show()


plt.hist(df['chol'], bins = 'auto', density = True, stacked = True, facecolor = "b", edgecolor="#6A9662")
plt.title('Total cholesterol Histogram')
plt.xlabel("serum cholesterol (mg/dl)")
plt.axvline(x=200, ymin=0, ymax=1, color='y') #borderline high
plt.axvline(x=239, ymin=0, ymax=1, color='#5c0e14', linewidth=4)    #high
plt.text(200, 0.009, "Borderline High value")
plt.text(300, 0.009, "High value",fontsize=18)
plt.show()

fbs_count = df['fbs'].value_counts()
plt.bar(('Normal','Not Normal'),(fbs_count[0],fbs_count[1]), width=0.5)
plt.title('fasting blood sugar Distribution')
plt.show()

restecg_count = df['restecg'].value_counts()
plt.bar(('Normal','having ST-T','Hypertrophy'),(restecg_count[0],restecg_count[1],restecg_count[2]), width=0.3)
plt.title('fresting electrocardiographic Distribution')
plt.show()



plt.hist(df['thalach'], bins = 'auto', facecolor = "b")
plt.title('Max Heart rate distribution')
plt.xlabel("maximum heart rate achieved")
plt.show()

exang_count = df['exang'].value_counts()
plt.bar(('No angina','Has Angina'),(exang_count[0],exang_count[1]), width=0.5)
plt.title('exercise induced angina Y/N')
plt.show()


plt.hist(df['oldpeak'], bins = 'auto', facecolor = "b")
plt.title('ST depression induced by exercise relative to rest distribution')
plt.xlabel("values")
plt.show()

slope_count = df['slope'].value_counts()
slope_types = ('upsloping','flat','downsloping')
plt.bar(slope_types,(slope_count[0],slope_count[1],slope_count[2]), width=0.3)
plt.title('the slope of the peak exercise ST segment')
plt.show()


vess_count = df['ca'].value_counts()
vessel_options = ('0','1','2','3','unknown')
plt.bar(vessel_options,(vess_count[0],vess_count[1],vess_count[2],vess_count[3],vess_count[4]), width=0.25)
plt.title('number of major vessels colored by flourosopy')
plt.show()


thal_count = df['thal'].value_counts()
thallium_options = ('missing value','fixed defect','normal','reversable defect')
plt.bar(thallium_options,(thal_count[0],thal_count[1],thal_count[2],thal_count[3]), width=0.25)
plt.title('Thallium scan results distribution')
plt.show()



ratio_y = round(sum(df['y'] == 1)/len(df),3)
plt.bar(['No','Yes'],(1-ratio_y,ratio_y), width=0.5)
plt.title('Heart attack probability')
plt.xlabel("Had Heart attack (Y/N)")
plt.show()

#print (corrMatrix)

#
plt.scatter(df['slope'], df['oldpeak'])
plt.title('Slope vs Oldpeak')
plt.xlabel('Slope')
plt.ylabel('Oldpeak')
plt.show()

plt.scatter(df['chol'], df['gender'])
plt.title('Cholesterol vs Gender')
plt.xlabel('Cholesterol')
plt.ylabel('Gender')
plt.show()

plt.scatter(df['talach'], df['age'])
plt.title('Talach vs Age')
plt.xlabel('Talach')
plt.ylabel('Age')
plt.show()

plt.scatter(df['y'], df['age'])
plt.title('Y vs Age')
plt.xlabel('Y')
plt.ylabel('Age')
plt.show()

#