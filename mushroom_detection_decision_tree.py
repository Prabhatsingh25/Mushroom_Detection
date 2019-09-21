
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree


# In[16]:


df = pd.read_csv("J:\Machine_Learning\mushroom_detection_decision_tree\mushrooms.csv")
df


# In[40]:


new_df = df.odor
new_df= pd.concat([new_df,df['gill-spacing'],df['stalk-surface-above-ring'],df['veil-color'],df['class']],axis='columns')
new_df


# In[41]:


le_odor = LabelEncoder()                        # making the encoder for 5 of the column which are more responsible for detection
le_gill = LabelEncoder()
le_stalk = LabelEncoder()
le_veil = LabelEncoder()
le_class = LabelEncoder()


# In[42]:


new_df['odor'] = le_odor.fit_transform(new_df['odor'])        # now adding the incoder first new column name 'odor' then encoder name 'le_odor' and the compair to which column of the data frame 'odor'
new_df['gill'] = le_gill.fit_transform(new_df['gill-spacing'])
new_df['stalk'] = le_stalk.fit_transform(new_df['stalk-surface-above-ring'])
new_df['veil'] = le_veil.fit_transform(new_df['veil-color'])
new_df['class'] = le_class.fit_transform(new_df['class'])
new_df = new_df.drop(['gill-spacing','stalk-surface-above-ring','veil-color'],axis='columns') # now drop all the text column
new_df 


# In[43]:


input_ = new_df.drop(['class'],axis='columns')             # now dividing the data in input and output
input_
output_ = new_df['class']
output_


# In[44]:


from sklearn.model_selection import train_test_split


# In[45]:


X_train, X_test, y_train, y_test = train_test_split(input_,output_,test_size = 0.2) # split the data in test and train


# In[46]:


len(X_train)


# In[47]:


tree_classification = tree.DecisionTreeClassifier()            # call dicision tree algo


# In[49]:


tree_classification.fit(X_train,y_train)                      # apply DecisionTreeClassifier


# In[51]:


tree_classification.score(X_test,y_test)                     # find the accuracy

