import pandas as pd
def train(df):
	ans=pd.DataFrame(columns=['attribute_name','attribute_value','class_name','class_value','count']);
	for i in df.columns:
    		for j in df[i].unique():
        		for k in df[df.columns[-1]].unique():
            			tmp=df.loc[df[i]==j].loc[df[df.columns[-1]]==k].shape[0];
            #print(str(i)+" "+str(j)+" "+str(df.columns[-1])+" "+str(k)+" "+str(df.loc[df[i]==j].loc[df[df.columns[-1]]==k].shape[0]));
            #ans.append({'attribute_name':i,'attribute_value':j,'class_name':df.columns[-1],'class_value':k,'count':tmp},ignore_index=True,)
            			ans.loc[len(ans)]=[i,j,df.columns[-1],k,tmp]
	return ans;
def predict(ans,sample):
    most_prob=0;
    max_prob=0.0;
    for k in ans['class_value'].unique():
        prod=1.0;
        prob_k=float(ans.loc[ans['attribute_name']==ans['class_name'].unique()[0],:].loc[ans['class_value']==k,:].loc[:,'count'].sum())/ans.loc[ans['attribute_name']==ans['class_name'].unique()[0],'count'].sum();
       # print(k+" "+str(prob_k));
        total=float(ans.loc[ans['attribute_name']==ans['class_name'].unique()[0],'count'].sum());
        for l in ans['attribute_name'].unique()[:-1]:
            num=float(ans.loc[ans['attribute_name']==l].loc[ans['attribute_value']==sample[l]].loc[ans['class_value']==k].loc[:,'count'].sum())/total;
        #    print(l+" "+str(num));
            prod=prod*((num)/prob_k);
        #    print(prod)
        if(prod>max_prob):
            max_prob=prod;
            most_prob=k;
    return most_prob,max_prob;

	
