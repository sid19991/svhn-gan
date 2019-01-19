import pandas as pd
from modules import train,predict # import train and predict modules
#training on weather dataset
df=pd.read_csv('weather.nominal.arff.csv');
ans=train(df);	#get the count of frequencies
rand_sample_1={'outlook':'rainy','temperature':'cool','humidity':'high','windy':True};
print("First sample for weather data is");
print(rand_sample_1);
[most_prob_class,max_prob]=predict(ans,rand_sample_1); #predict the class
print("Most probable class for sample 1 is \'"+most_prob_class+"\' with maximum probability "+str(max_prob));
rand_sample_2={'outlook':'overcast','temperature':'mild','humidity':'normal','windy':False}
print("second sample for weather data is");
print(rand_sample_2);
[most_prob_2,max_prob_2]=predict(ans,rand_sample_2)
print("Most probable class for sample 2 is \'"+most_prob_2+"\' with maximum probability "+str(max_prob_2));
#training on second dataset (train.csv)
df=pd.read_csv('train_data.csv');
ans=train(df);
rand_sample_train1={'Day':'Holiday','Season':'Autumn','Wind':'High','Rain':'Slight'}
print("First sample for train data is");
print(rand_sample_train1);
[most_prob_1,max_prob_1]=predict(ans,rand_sample_train1)
print("Most probable class for sample 1 is \'"+most_prob_1+"\' with maximum probability "+str(max_prob_1));
rand_sample_train2={'Day':'Sunday','Season':'Winter','Wind':'High','Rain':'None'}
print("Second sample for weather data is");

print(rand_sample_train2);
[most_prob_2,max_prob_2]=predict(ans,rand_sample_train2)
print("Most probable class for sample 2 is \'"+most_prob_2+"\' with maximum probability "+str(max_prob_2));

