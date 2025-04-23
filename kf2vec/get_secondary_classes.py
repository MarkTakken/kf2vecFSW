
# Python script to get second/third/fourth best classification class

#Invoke with: python get_secondary_classes.py classes.out
#Invoke with: python get_secondary_classes.py /Users/nora/Documents/ml_metagenomics/tol_variable_k_resuls/10k_tol_good_chunked_genomes_queries_ONT_reads_acc0.95/k7_v38_8k_s28_clade_All_ONT_Reads_Good_Subset_acc0.95_ChunkModel/classes.out



import os
import sys
import pandas as pd
import numpy as np


inFile = sys.argv[1]
#outFile = sys.argv[2]

first_fi = os.path.basename(inFile)
p1 = os.path.dirname(inFile)


second_fi = "classes_secondBest.out"
third_fi = "classes_thirdBest.out"
fourth_fi = "classes_fourthBest.out"


classes_df1 = pd.read_csv(os.path.join(p1, first_fi), sep = "\t", header = 0)
L = np.argsort(-classes_df1.iloc[:, 3:].to_numpy(), axis=1)

df_classes_header = pd.DataFrame(columns=list(classes_df1.columns))


if os.path.isfile(os.path.join(p1, second_fi)):
    os.remove(os.path.join(p1, second_fi))
    df_classes_header.to_csv(os.path.join(p1, second_fi), index=False, sep='\t', mode = "a")
    
              
if os.path.isfile(os.path.join(p1, third_fi)):
    os.remove(os.path.join(p1, third_fi))
    df_classes_header.to_csv(os.path.join(p1, third_fi), index=False, sep='\t', mode = "a")

              
if os.path.isfile(os.path.join(p1, fourth_fi)):
    os.remove(os.path.join(p1, fourth_fi))
    df_classes_header.to_csv(os.path.join(p1, fourth_fi), index=False, sep='\t', mode = "a")



for i in range (0, len(classes_df1)):

   
    tmp_df = classes_df1.iloc[[i]]
    
    #print(tmp_df)
    
    first_best = L[i, 0]
    sec_best = L[i, 1]
    thi_best = L[i, 2]
    four_best = L[i, 3]

     # Output second best class
    tmp_df.loc[i, 'top_class'] = float(sec_best)
    tmp_df.loc[i, "top_p"] = classes_df1.iat[i, int(sec_best+3)]
    #print(tmp_df)
    tmp_df.to_csv(os.path.join(p1, second_fi), index=False, sep='\t', mode = "a", header = False)

     # Output third best class
    tmp_df.loc[i, 'top_class'] = float(thi_best)
    tmp_df.loc[i, "top_p"] = classes_df1.iat[i, int(thi_best+3)]
    #print(tmp_df)
    tmp_df.to_csv(os.path.join(p1, third_fi), index=False, sep='\t', mode = "a", header = False)

     # Output fourth best class
    tmp_df.loc[i, 'top_class'] = float(four_best)
    tmp_df.loc[i, "top_p"] = classes_df1.iat[i, int(four_best+3)]
    #print(tmp_df)
    tmp_df.to_csv(os.path.join(p1, fourth_fi), index=False, sep='\t', mode = "a", header = False)

    # break
    # assert(False)


