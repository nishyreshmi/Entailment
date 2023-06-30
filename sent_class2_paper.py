from snli_exp import exp1
from snli_exp import exp2

from nltk import pos_tag
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import re
import csv
#---------to extract sentences from SNLI to list.....
def extract_text():
    with open("snli_1.0_dev.txt","r") as data:
        train = csv.DictReader(data, delimiter='\t')
        evi_sentences = []
        hyp_sentences = []
        labels = []
        #scores = []
        for row in train:
            hyp_sentences.append(row["sentence1"])
            evi_sentences.append(row["sentence2"])
            labels.append(row["gold_label"])
    return hyp_sentences, evi_sentences, labels

hyp, evi, lab = extract_text()

#...adding no for incorrect n no labels....

lab_correct=[]
cor="yes"
not_cor="no"
avoid=[21,47,128,165,184,188,196,250,265,274,286,331,369,441,481,
       771,842,1055,1087,1107,1113,1213,1568,2042,2222,2514,2532,
       2717,2857,2989,3009,3046,3643,3877,
       4025,4200,4207,4732,4967,5037,5049,5628,5723,5801,5805,5864,
       5877,6288,6617,6670,7006,7113,7309,7427,7559,7586,7629,7675,
       7718,7941,8010,8812,8838,9993]
cc_avoid=[28,29,52,183,674,1024,1064,1103,1631,1700,2098,2160,
          2186,2352,2374,2706,2812,2982,3035,3053,3075,4022,4030,
          4062,4380,4404,4717,4990,5024,5175,5421,5436,5662,5720,
          5769,5792,5973,6467,6513,6520,6524,6560,6622,6725,6732,
          7628,7735,7894,7919,8194,8500,8538,8590,8612,8705,9047,
          9101,9167,9233,9336,9681,9686,9725,9824,]
ee_avoid=[30,62,70,132,182,191,192,222,235,366,389,416,438,480,508,
          574,681,689690,737,759,772,780,837,853,875,918,919,973,
          982,991,1028,1045,1129,1210,1262,1288,1437,1487,1490,
          1502,1504,1517,1579,1597,1625,1626,1628,1676,1711,1739,
          1768,1807,1888,1917,1963,1973,1985,2005,2035,2040,2058,
          2071,2075,2157,2184,2195,2198,2221,2231,2241,2244,2247,
          2382,2443,2449,2479,2493,2565,2568, 2571,2573,2581,2589,
          2642,2644,2710,2724,2765,2821,2824,2836,2846,2852,2861,
          2867,2893,2906,2932,2959,3049,3077,3087,3116,3120,3141,
          3146,3160,3182,3192,3202,3206,3213,3264,3265,3307,3316,
          3385,3393,3416,3429,3434,3450,3459,3478,3482,3491,3505,
          3518,3536,3554,3564,3568,3574,3631,3642,3654,3671,3685,
          3692,3696,3710,3720,3732,3734,3743,3752,3792,3799,3805,
          3809,3868,3869,3881,3948,3960,3982,4002,4069,4073,4080,
          4114,4138,4145,4189,4205,4220,4225,4245,4259,4266,4280,
          4290,4319,4324,4350,4357,4368,4370,4374,4381,4394,4406,
          4469,4503,4527,4533,4545,4548,4657,4669,4722,4747,4765,
          4785,4832,4838,4867,4883,4884,4905,4923,4924,4944,4965,
          4978,4984,4987,4993,5011,5014,5032,5050,5069,5108,5114,
          5137,5153,5178,5185,5192,5225,5229,5231,5258,5267,5284,
          5311,5314,5334,5351,5382,5383,5390,5393,5413,5428,5435,
          5457,5461,5465,5474,5476,5481,5497,5554,5572,5579,5582,
          5585,5590,5593,5596,5599,5635,5650,5652,5659,5666,5667,
          5673,5739,5747,5762,5767,5847,5899,5919,5920,5925,5942,
          5953,5960,5970,5972,5990,5995,5997,5999,6068,6089,6090,
          6103,6109,6116,6118,6135,6145,6149,6155,6158,6173,6175,
          6194,6207,6210,6211,6222,6229,6259,6267,6307,6313,6328,
          6395,6396,6432,6440,6465,6474,6475,6479,6488,6517,6536,
          6587,4801,6802,6805,6860,6889,6898,6964,7030,7034,7042,
          7059,7079,7109,7115,7159,7174,7220,7235,7248,7260,7266,
          7291,7320,7390,7409,7435,7458,7496,7526,7536,7546,7550,
          7578,7594,7631,7641,7680,7729,7783,7810,7835,7838,7863,
          7870,7875,7878,7881,7882,7905,7911,7949,7977,7987,7996,
          8011,8020,8023,8072,8079,8136,8139,8150,8167,8188,8191,
          8195,8208,8213,8215,8275,8285,8310,8313,8318,8339,8353,
          8416,8426, 8447,8496,8539,8540,8581,8622,8662,8711,8725,
          8747,8753,8757,8791,8802,8939,8956,8998,9020,9023,9053,
          9085,9203,9216,9218,9273,9275,9279,9343,9410,9423,
          9485,9514,9515,9521,9575,9586,9587,9610,9638,9645,
          9672,9678,9708,9710,9737,9744,9766,9788,9789,9833,9872,
          9923,9942,9979]
        
pars_incor=[115,250,300,318,354,418,453,490,491,503,585,616,622,
            666,693,898,1011,1134,1145,1150,1168,1181,1226,1306,
            1372,1465,1549,1550,1551,1689,1705,1733,
            1778,1781,1811,1852,1924,1926,2012,2152,2167,2170,2217,
            2276,2253,
            2277,2514,2532,2554,2580,2694,2761,2790,2803,2855,2865,2897,2985,3095,3111,3200,3201,3268,3271,3423,
            3572,3628,3630,3723,4137,4190,4251,4270,4341,4390,4456,4604,4767,4793,4900,4999,
            5067,5072,5121,5128,5134,5164,5205,5304,5419,5420,5501,6418,6640,6645,6704,6981,7011,7199,7745,7755,7918,8049,
            8101,8220,8268,8346,8432,8635,8645,8646,8921,9156,9174,
            9234,9888]
sent_incor=[24,25,26,33,35,56,73,74,82,96,97,98,102,103,105,106,138,139,140,177,214, 228,
            244, 255, 256, 257, 306, 307, 308, 323, 324, 325, 326, 327, 329, 330,
            339, 340,341,380,402,403, 404, 442,443, 454,504,510,511,519,520,521,
            558,559,560,561,562,563,597,598,599,612,613,623,836,636,637,638,639,640,
            660,669,671,696,697,698,722, 755, 766, 767, 768, 784, 790, 791, 850,
            851,852, 874, 876, 877, 879, 887, 888, 895, 896, 897, 922, 928, 938,
            946, 947, 948, 952, 953, 954, 957, 958, 959, 960, 980, 981,1014, 1033,
            1034, 1035, 1054, 1056, 1085, 1096, 1098,
            1149, 1153, 1161, 1162, 1163, 1164, 1180, 1182, 1191, 1198, 1199, 1200,
            1216, 1218, 1230, 1258, 1269, 1276, 1286, 1303, 1309, 1311, 1327, 1328,
            1329, 1353, 1370, 1379, 1392, 1394, 1395, 1423, 1424, 1425, 1432, 1433, 1434,
            1434, 1476, 1495, 1496, 1497, 1498, 1499,
            1559, 1560, 1596, 1623, 1644, 1654, 1660, 1661, 1662, 1663,1664,1665,
            1712, 1735, 1736, 1737, 1738, 1747, 1748, 1749, 1774, 1775, 1776, 1777, 1790, 1792,
            1825, 1826, 1827, 1840, 1841, 1842, 1854, 1882, 1883,
            1942, 1943, 1944, 1981, 1982, 1983, 1987, 1988,2002,2003,2004,
            2033,2047,2048,2049,2065,2066,2067,2104, 2105, 2106, 2108, 2143, 2144, 2145,
            2161,2162,2163,2173,2177,2181,2192,
            2220, 2224, 2225, 2226, 2236, 2238, 2260, 2261, 2262, 2292, 2299,
            2300, 2301, 2314, 2315, 2316, 2333, 2338, 2339, 2340, 2343, 2347,
            2385, 2386, 2387, 2388,2401, 2402, 2403, 2420, 2446, 2453, 2454, 2464, 2466, 2470, 2471, 2472, 2494, 2495, 2496, 2498,
            2506, 2507, 2508,2647, 2648, 2649, 2656, 2657, 2658, 2668, 2670, 2671,2672, 2673, 2679, 2684, 2685, 2690, 2691, 2695, 2696, 2697,
            2713, 2714, 2715, 2725, 2726, 2727, 2755, 2767, 2769,2800, 2828, 2829, 2831, 2832, 2834, 2840, 2899,
            2900, 2921, 2922, 2943, 2986, 2988, 2994,
            3000, 3023, 3028, 3029, 3030, 3037, 3038, 3039, 3058, 3059, 3128, 3171, 3173, 3181, 3183, 3185,
            3220, 3238, 3239, 3240, 3241, 3242, 3243, 3284, 3285, 3286, 3300, 3313, 3314, 3315, 3347, 3348, 3355, 3356, 3357,3363, 3376, 3377,
            3378, 3381,3409, 3418,3454, 3455, 3456, 3460,
            3538, 3539, 3540, 3544, 3559, 3591, 3595, 3596, 3601, 3602, 3613, 3614, 3615, 3616, 3617, 3623, 3624, 3625, 3626, 3627, 3682, 3683,
            3748, 3749, 3750,3763, 3815,
            3816, 3820, 3821, 3822, 3826, 3828, 3829, 3830, 3831, 3839,3866, 3882,
            3904, 3905, 3906, 3920, 3921, 3925, 3926, 3927, 3931, 3940, 3941, 3942, 3943, 3945, 3967, 3968, 3969, 3977, 3984,
            3985, 3986, 3987, 3988, 3989, 3990,
            4009, 4010, 4011, 4015, 4016, 4017, 4027, 4037, 4039, 4040, 4041, 4048, 4049, 4050, 4075, 4076, 4077,4099, 4100,
            4123, 4174, 4175, 4182, 4192, 4193, 4194, 4216, 4217, 4218, 4246, 4247, 4248, 4255,4258, 4276, 4277, 4278,

            4309, 4321, 4322, 4323, 4351, 4352, 4353, 4354, 4355, 4356, 4382, 4392, 4411, 4420, 4421, 4427, 4428,4457, 4428, 4442, 4447, 4449,
            4462, 4463, 4464, 4465, 4492, 4493,4494,4510, 4511, 4512, 4513, 4514, 4515, 4534, 4535, 4536, 4537, 4538, 4539, 4576, 4577,
            4615, 4616, 4617, 4645, 4646, 4647, 4648, 4649, 4675, 4677, 4698, 4726, 4727, 4728, 4746, 4791, 4821, 4858, 4859, 4860, 4861,
            4861, 4862, 4863, 4879, 4880, 4881, 4887, 4889, 4890, 4894, 4895,4896, 4898, 4928, 4929, 4934, 4962,4972, 4973, 4974,
            5044, 5045, 5046, 5079, 5086, 5089, 5090, 5091, 5112, 5116, 5117, 5118, 5150, 5184, 5234, 5272, 5273, 5274, 5275, 5276, 5277, 5281, 5283, 5308,
            5309, 5310, 5316, 5320, 5322, 5331,
            5341, 5342, 5343, 5365, 5366, 5392, 5394, 5398, 5398, 5399, 5399, 5400, 5400, 5408, 5409, 5410, 5411,5433,5434, 5412, 5417, 5418,5440, 5441, 5442,
            5452, 5453, 5454, 5469, 5470, 5471, 5472, 5494, 5495, 
            5524, 5525, 5526,5588, 5605, 5606, 5607, 5623, 5624, 5625, 5648, 5656, 5658, 5668, 5669, 5713, 5713, 5714, 5714, 5715, 5716, 5718, 5743, 5744, 5745,
            5750, 5758, 5798,5799, 5815, 5816, 5817, 5818, 5819, 5820, 5821, 5822,5941,5943,5944,5945,5946,5991,
            6000, 6010, 6018, 6020, 6021, 6034, 6035, 6036, 6053, 6056, 6057, 6074, 6085, 6086, 6087, 6094, 6095, 6096, 6130, 6132,
            6136, 6137, 6165, 6167, 6168, 6169, 6170, 6197, 6232, 6233, 6234, 6244,  6245, 6246, 6278, 6279, 6292, 6299,
            6309, 6310, 6311, 6321, 6331, 6339, 6348, 6397, 6398, 6399, 6406, 6408, 6436, 6437, 6438, 6445, 6446, 6447, 6448, 6449, 6450,
            6487,6489, 6490, 6491, 6492,6589, 6590, 6591, 6592, 6626, 6627, 6628, 6690, 6707, 6708, 6718, 6719, 6720, 6726, 6755, 6778, 6779, 6780, 6808, 6809,6810, 6811, 6812,
            6813, 6817, 6818, 6819, 6844, 6845, 6850, 6851, 6852, 6880, 6881, 6882, 6903, 6918, 6920, 6921, 6944, 6967, 968, 6969, 6970,
            6971, 6972,7000, 7001, 7002, 7015, 7016, 7017, 7036, 7037, 7039, 7040, 7048, 7049, 7050, 7098, 7198, 7222, 7223, 7224, 7244, 7267, 7303, 7304,
            7305, 7333, 7335,7336, 7337, 7338, 7372, 7373, 7375, 7376, 7376, 7377, 7384, 7385, 7386, 7387, 7389, 7393, 7394, 7395, 7396, 7397, 7402, 7420,
            7421, 7422, 7453, 7454, 7455, 7469,7473, 7475, 7476, 7478, 7485, 7507, 7508, 7509, 7509, 7516, 7517, 7554, 7588, 7589, 7590, 7597, 7598, 7599,
            7657, 7672, 7672, 7673,  7674, 7696, 7697, 7699, 7700,7726, 7727, 7728, 7733, 7734, 7768, 7769, 7770, 7774, 7775, 7776, 7798, 7799, 7800, 7813,
            7814, 7815, 7822, 7852, 7854, 7896, 7900, 7901, 7902, 7928, 7930, 7933,7934, 7935, 7936, 7938,7951, 7957, 7958, 7959, 7989,
            8008, 8015, 8016, 8032, 8050, 8052, 8053, 8054, 8055, 8098, 8099, 8105, 8106, 8110, 8122, 8124,
            8147, 8148, 8181, 8185, 8197, 8198, 8199,8224, 8225, 8226, 8227, 8230, 8231, 8240, 8278, 8279, 8280, 8287, 8289, 8293, 8294, 8295, 8296, 8297,
            8298, 8299, 8300, 8301, 8328, 8347, 8350, 8398, 8399, 8476, 8477, 8478,8498, 8499,8503, 8504, 8505, 8509, 8512, 8513, 8514, 8515, 8516, 8521, 8522,
            8523, 8578, 8579, 8580, 8593, 8594, 8595, 8605, 8606, 8607, 8609, 8626, 8627, 8628,8628, 8663, 8674,  8675, 8676, 8695, 8696, 8773, 8774, 8775,
            8804, 8845, 8845, 8846, 8847, 8851, 8852, 8853,8879, 8886, 8887, 8888, 8889,8909, 8914, 8915, 8916, 8929, 8931, 8941, 8942, 8943, 8944, 8945,
            8946, 8947, 8948, 8949, 8974,8980, 8989, 8990, 8991,9168,9011, 9058, 9059, 9060, 9064, 9065, 9066, 9091, 9092, 9093, 9097, 9098, 9099, 9118, 9119, 9120,
            9121, 9122, 9123, 9137, 9149,9220, 9221, 9222, 9223,
            9224, 9225, 9253, 9254, 9268, 9268, 9269, 9269, 9270,
            9286, 9287, 9288, 9292, 9298, 9313, 9314, 9315, 9316, 9316, 9317, 9318,
            9337, 9338, 9339, 9364, 9365, 9366, 9391, 9392, 9393, 9411, 9419, 9433, 9434, 9435, 9442, 9443, 9443, 9444, 9469, 9481, 9482, 9493, 9494, 9495,
            9496,9517, 9518, 9519, 9526, 9527, 9528, 9534, 9539,
            9553, 9553, 9554, 9554, 9555, 9607, 9608, 9608, 9609,
            9614, 9630, 9631, 9632, 9633, 9652,
            9658, 9659,9660, 9661, 9662, 9663, 9682, 9682, 9683,
            9683, 9684, 9684, 9703, 9704, 9705, 9721, 9723, 9727,
            9728, 9729, 9734, 9739, 9754, 9769,
            9796, 9797, 9805, 9806,9807, 9814, 9815, 9816, 9822, 9844, 9845, 9893, 9894, 9917, 9926, 9927, 9945, 9997]

for i in range(10000):
    if ((lab[i] =="-") or(i in avoid) or(i in ee_avoid)or (i in cc_avoid)or(i in sent_incor) or (i in pars_incor)):# (i==184) or (i==188 )or(i== 196)or (i==128)or(i==47)or(i==274) or(i==286)or(i==331):
       lab_correct.append(not_cor)
    else :
        lab_correct.append(cor)

count=0
for i in range(1000):
    if lab_correct[i]=='no':
               count+=1

l=9000
u=10000
hyp_sample=hyp[l:u]
evi_sample=evi[l:u]
lab_sample=lab[l:u]
lab_cor_sample=lab_correct[l:u]


colours=["black","blue","white","red","green","orange","brown","yellow","pink","green","purple"]

cnt_e=0
cnt_c=0
cnt_n=0

countc=0
counte=0
countn=0
#exp1....   
from nltk.stem.wordnet import WordNetLemmatizer
incor_sent=[]
for p in range(len(hyp_sample)):
   if (lab_cor_sample[p]=='yes'):
    hyp_sen1=hyp_sample[p]
 
    #------------tokens,stop word ,punctuation removal,lemmatize....
    lm = WordNetLemmatizer()
    ps = PorterStemmer()
    hyp_tok=[]
    hyp_tok1=[]
   
    
    hyp_sen2=hyp_sen1.split("-")
    hyp_sen= " " .join(hyp_sen2)
    hyp_input_sen=hyp_sen
    hyp_tk=word_tokenize(hyp_input_sen.lower())
    hyp_tok1=pos_tag(word_tokenize(hyp_input_sen))
    ps=['VBZ','WRB','VBP']
    pp=['VBD','VBN']
    nn=['NNS','NNPS']
    nn_exclude=["building","ceiling","king","ping","ring","singer","something",'spring',"thing","wing"]
    for i in range(len(hyp_tok1)):
      if (hyp_tok1[i][1] in nn):
          x=exp1.stm_nns(hyp_tok1[i][0].lower())
          if x==None:
                  x=hyp_tok1[i][0].lower()

          hyp_tok.append(x)
##          # # print (x)
      elif (hyp_tok1[i][1]=='VBG'):
        x=exp1.stm_vbg(hyp_tok1[i][0].lower())
        if x==None:
                  x=hyp_tok1[i][0].lower()

        hyp_tok.append(x)
##        # # print (x)
      elif (hyp_tok1[i][1] in ps):
        x=exp1.stm_vbz(hyp_tok1[i][0].lower())
        if x==None:
                  x=hyp_tok1[i][0].lower()
        hyp_tok.append(x)
##        # # print (x)
      elif (hyp_tok1[i][1] in pp):
        x=exp1.stm_vbd(hyp_tok1[i][0].lower())
        if x==None:
                  x=hyp_tok1[i][0].lower()
        hyp_tok.append(x)
##        # # print (x)
      elif (hyp_tok1[i][1] =="NN" and re.search("ing",hyp_tok1[i][0]) and (hyp_tok1[i][0] not in nn_exclude)):
            x=exp1.stm_vbg(hyp_tok1[i][0])
            if x==None:
                x=hyp_tok1[i][0].lower()
            hyp_tok.append(x)
      else:
        
        x=hyp_tok1[i][0].lower()
        hyp_tok.append(x)
##        # # print (x)

    hyp_filt_sent=exp1.stop_wd(hyp_tok)
    hyp_lem1=exp1.elim_pun(hyp_filt_sent)



    
   #---------------------parsing
    

    hyp_triples,hyp_pos=exp1.extract_triples(hyp_sen)
    for i in range(len(hyp_triples)):
        if hyp_triples[i][2]=='cop':

            hyp_sub1,hyp_ver1,hyp_obj1,h_sub1,h_ver1,h_obj1=exp1.extract_cop(hyp_triples,hyp_pos)
            hyp_subnmod1,hyp_vernmod1,hyp_objnmod1,hyp_subnmod,hyp_vernmod,hyp_objnmod=exp1.extract_parts_nmod(hyp_triples,hyp_obj1,hyp_ver1,hyp_sub1,hyp_pos)
            break 

        elif hyp_triples[i][2]=='nsubjpass':
            
               hyp_sub1,hyp_ver1,hyp_obj1,h_sub1,h_ver1,h_obj1=exp1.extract_pass(hyp_triples,hyp_pos)
               if (hyp_obj1==[]):

                 hyp_subnmod1,hyp_vernmod1,hyp_objnmod1,hyp_subnmod,hyp_vernmod,hyp_objnmod=exp1.extract_parts_nmod(hyp_triples,h_obj1,h_ver1,h_sub1,hyp_pos)
                 if(hyp_vernmod1): 

                    h_obj1.append(hyp_vernmod1[0])
                    w_pos=exp1.extract_pos(hyp_pos,hyp_vernmod1[0])

                    if w_pos=="NNS" or "NNPS":
                      w=exp1.stm_nns(hyp_vernmod1[0])
                    elif w_pos=="NN":
                        w=hyp_vernmod1[0].lower()

                    if w==None:
                         w=hyp_vernmod1[0].lower()
                    if w not in hyp_obj1:
             
                            hyp_obj1.append(w)
  
               hyp_subnmod1,hyp_vernmod1,hyp_objnmod1,hyp_subnmod,hyp_vernmod,hyp_objnmod=exp1.extract_parts_nmod(hyp_triples,h_obj1,h_ver1,h_sub1,hyp_pos)
                  
                  
               break    

        elif (hyp_triples[i][2]=='nsubj' or hyp_triples[i][2]=='acl'):
 
                hyp_sub1,hyp_ver1,hyp_obj1,h_sub1,h_ver1,h_obj1=exp1.extract_parts_svo(hyp_triples,hyp_pos)

                hyp_subnmod1,hyp_vernmod1,hyp_objnmod1,hyp_subnmod,hyp_vernmod,hyp_objnmod=exp1.extract_parts_nmod(hyp_triples,h_obj1,h_ver1,h_sub1,hyp_pos)
                break
        else:
            hyp_sub1=[]
            h_sub1=[]
            hyp_ver1=[]
            h_ver1=[]
            hyp_obj1=[]
            h_obj1=[]
            hyp_subnmod1=[]
            hyp_vernmod1=[]
            hyp_objnmod1=[]
            
    if (hyp_sub1==hyp_ver1==hyp_obj1==[]):        

       incor_sent.append(p)
    animal=["dog","cat","horse","cow","elephant"]
    costum=['shirt','costume','dress','skirt','suit','coat','outfit','jersey','top','pants','suits','jacket','trunk','hoodie','sweatshirt','bathrobe','clothing','sandal']
    if((hyp_sub1) and (hyp_subnmod1)):
        for i in hyp_subnmod1:
          if (i in costum):
               hyp_st=exp1.in_correction(h_sub1,hyp_tk)
               hyp_triples,hyp_pos=exp1.extract_triples(hyp_st)

               hyp_tok_new=[]
               hyp_tok11=pos_tag(word_tokenize(hyp_st))
               ps=['VBZ','VBP','WRB']
               pp=['VBD','VBN']
               for i in range(len(hyp_tok11)):
                  if (hyp_tok11[i][1] in nn):
                     x=exp1.stm_nns(hyp_tok11[i][0].lower())
                     if x==None:
                         x=hyp_tok11[i][0].lower()
                     hyp_tok_new.append(x)

                  elif (hyp_tok11[i][1]=='VBG'):
                     x=exp1.stm_vbg(hyp_tok11[i][0].lower())
                     if x==None:
                         x=hyp_tok11[i][0].lower()
                     hyp_tok_new.append(x)

                  elif (hyp_tok11[i][1] in ps):

                     x=exp1.stm_vbz(hyp_tok11[i][0].lower())
                     if x==None:
                           x=hyp_tok11[i][0].lower()
                     hyp_tok_new.append(x)

                  elif (hyp_tok11[i][1] in pp):
                    x=exp1.stm_vbd(hyp_tok11[i][0].lower())
                    if x==None:
                        x=hyp_tok11[i][0].lower()
                    hyp_tok_new.append(x)

                  elif (hyp_tok11[i][1] =="NN" and re.search("ing",hyp_tok11[i][0])and (hyp_tok11[i][0] not in nn_exclude)):
                      x=exp1.stm_vbg(hyp_tok11[i][0])
                      if x==None:
                         x=hyp_tok11[i][0].lower()
                      hyp_tok_new.append(x)
                  else:
        
                     x=hyp_tok11[i][0].lower()
                     hyp_tok_new.append(x)
               hyp_filt_sent1=exp1.stop_wd(hyp_tok_new)
               hyp_lem1=exp1.elim_pun(hyp_filt_sent1)
 

    
               hyp_sub1,hyp_ver1,hyp_obj1,h_sub1,h_ver1,h_obj1=exp1.extract_parts_svo(hyp_triples,hyp_pos)


               hyp_subnmod1,hyp_vernmod1,hyp_objnmod1,hyp_subnmod,hyp_vernmod,hyp_objnmod=exp1.extract_parts_nmod(hyp_triples,h_obj1,h_ver1,h_sub1,hyp_pos)

    hyp_color_amod,hyp_mod=exp1.extract_parts_mod(hyp_triples,hyp_obj1,hyp_pos,hyp_subnmod1,hyp_vernmod1,colours)

    hyp_mark,hyp_advcl,hyp_advmod,hyp_neg,hyp_prt=exp1.extract_parts_other(hyp_triples,hyp_pos)

    hyp_subcomp,hyp_objcomp,hyp_sncomp,hyp_vncomp,hyp_oncomp=exp1.extract_parts_comp(hyp_triples,h_obj1,h_ver1,h_sub1,hyp_subnmod1,hyp_vernmod1,hyp_objnmod1,hyp_pos)
    hyp_subnum,hyp_objnum=exp1.extract_parts_num(hyp_triples,h_obj1,h_sub1,hyp_pos)
    hyp_subconj,hyp_objconj=exp1.extract_parts_conj(hyp_triples,h_obj1,h_sub1,hyp_pos)

    for i in hyp_subconj:
        hyp_sub1.append(i)
    for i in hyp_objconj:
        hyp_obj1.append(i)
    
    for i in hyp_subcomp:
        hyp_sub1.append(i)
    for i in hyp_objcomp:
        hyp_obj1.append(i)

    h_ver1=exp1.is_convert(h_ver1)

    gr=["group","bunch","pair"]
    for i in h_sub1:
      if i in gr:
         if(hyp_subnmod1):
            hyp_sub1.append(hyp_subnmod1[0])

    lmtzr = WordNetLemmatizer()
    evi_tok=[]
    evi_sen1=evi_sample[p]

    
    evi_sen2=evi_sen1.split("-")
    evi_sen= " " . join(evi_sen2)
    evi_input_sen=evi_sen
    evi_tk=word_tokenize(evi_input_sen.lower())
    evi_tok1=pos_tag(word_tokenize(evi_input_sen))

    ps=['VBZ','VBP','WRB']
    pp=['VBD','VBN']
    nn=['NNS','NNPS']
    for i in range(len(evi_tok1)):
      if (evi_tok1[i][1] in nn):
          x=exp1.stm_nns(evi_tok1[i][0].lower())
          if x==None:
                  x=evi_tok1[i][0].lower()
          evi_tok.append(x)

      elif (evi_tok1[i][1]=='VBG'):
        x=exp1.stm_vbg(evi_tok1[i][0].lower())
        if x==None:
                  x=evi_tok1[i][0].lower()
        evi_tok.append(x)

      elif (evi_tok1[i][1] in ps):
        x=exp1.stm_vbz(evi_tok1[i][0].lower())
        if x==None:
                  x=evi_tok1[i][0].lower()
        evi_tok.append(x)

      elif (evi_tok1[i][1] in pp):
        x=exp1.stm_vbd(evi_tok1[i][0].lower())
        if x==None:
                  x=evi_tok1[i][0].lower()
        evi_tok.append(x)

      elif (evi_tok1[i][1] =="NN" and re.search("ing",evi_tok1[i][0]) and (evi_tok1[i][0] not in nn_exclude)):
         x=exp1.stm_vbg(evi_tok1[i][0])
         if x==None:
            x=evi_tok1[i][0].lower()
         evi_tok.append(x)
      else:
        
        x=evi_tok1[i][0].lower()
        evi_tok.append(x)
    evi_filt_sent=exp1.stop_wd(evi_tok)
    evi_lem1=exp1.elim_pun(evi_filt_sent)

    #---------------------parsing
    evi_triples,evi_pos=exp1.extract_triples(evi_sen)
 
    for i in range(len(evi_triples)):
        if evi_triples[i][2]=='cop':

            evi_sub1,evi_ver1,evi_obj1,e_sub1,e_ver1,e_obj1=exp1.extract_cop(evi_triples,evi_pos)
            evi_subnmod1,evi_vernmod1,evi_objnmod1,evi_subnmod,evi_vernmod,evi_objnmod=exp1.extract_parts_nmod(evi_triples,e_obj1,e_ver1,e_sub1,evi_pos)
            break

        elif evi_triples[i][2]=='nsubjpass':
#.............taking nsubjpass as subj............
               evi_sub1,evi_ver1,evi_obj1,e_sub1,e_ver1,e_obj1=exp1.extract_pass(evi_triples,evi_pos)
               if(evi_obj1==[]):

                 evi_subnmod1,evi_vernmod1,evi_objnmod1,evi_subnmod,evi_vernmod,evi_objnmod=exp1.extract_parts_nmod(evi_triples,e_obj1,e_ver1,e_sub1,evi_pos)
                 if (evi_vernmod1):

                     e_obj1.append(evi_vernmod1[0]) 
                     w_pos=exp1.extract_pos(evi_pos,evi_vernmod1[0])

                     if w_pos=="NNS" or "NNPS":
                       w=exp1.stm_nns(evi_vernmod1[0])
                     elif w_pos=="NN":
                        w=evi_vernmod1[0].lower()

                     if w==None:
                         w=evi_vernmod1[0].lower()
                     if w not in evi_obj1:
             
                            evi_obj1.append(w)

                   


               evi_subnmod1,evi_vernmod1,evi_objnmod1,evi_subnmod,evi_vernmod,evi_objnmod=exp1.extract_parts_nmod(evi_triples,e_obj1,e_ver1,e_sub1,evi_pos)
               break
        elif ((evi_triples[i][2]=='nsubj')or (evi_triples[i][2]=='acl')):   
                 evi_sub1,evi_ver1,evi_obj1,e_sub1,e_ver1,e_obj1=exp1.extract_parts_svo(evi_triples,evi_pos)

                 evi_subnmod1,evi_vernmod1,evi_objnmod1,evi_subnmod,evi_vernmod,evi_objnmod=exp1.extract_parts_nmod(evi_triples,e_obj1,e_ver1,e_sub1,evi_pos)
                 break
        else:
            evi_sub1=[]
            e_sub1=[]
            evi_ver1=[]
            e_ver1=[]
            evi_obj1=[]
            e_obj1=[]
            evi_subnmod1=[]
            evi_vernmod1=[]
            evi_objnmod1=[]
       
    if (evi_sub1==evi_ver1==evi_obj1==[]):        

       incor_sent.append(p)
     
    costum=['shirt',"short",'costume','dress','skirt','suit','outfit','coat','jersey','top','pants','suits','jacket',
            'trunk','hoodie','sweatshirt','bathrobe','clothing',"vest","tutu"]
    if((evi_sub1) and (evi_subnmod1)):
 
      for i in evi_subnmod1:
        if (i in costum):
        
        
        
        
          evi_st=exp1.in_correction(e_sub1,evi_tk)
          evi_triples,evi_pos=exp1.extract_triples(evi_st)

          evi_tok_new=[]
          evi_tok11=pos_tag(word_tokenize(evi_st))

          ps=['VBZ','VBP','WRB']
          pp=['VBD','VBN']
          for i in range(len(evi_tok11)):
             if (evi_tok11[i][1] in nn):
                x=exp1.stm_nns(evi_tok11[i][0].lower())
                if x==None:
                  x=evi_tok11[i][0].lower()
                evi_tok_new.append(x)

             elif (evi_tok11[i][1]=='VBG'):
                x=exp1.stm_vbg(evi_tok11[i][0])
                if x==None:
                  x=evi_tok11[i][0].lower()
                evi_tok_new.append(x)

             elif (evi_tok11[i][1] in ps):
                x=exp1.stm_vbz(evi_tok11[i][0].lower())
                if x==None:
                  x=evi_tok11[i][0].lower()
                evi_tok_new.append(x)

             elif (evi_tok11[i][1] in pp):
                x=exp1.stm_vbd(evi_tok11[i][0].lower())
                if x==None:
                  x=evi_tok11[i][0].lower()
                evi_tok_new.append(x)

             elif (evi_tok11[i][1] =="NN" and re.search("ing",evi_tok11[i][0])and (evi_tok11[i][0] not in nn_exclude)):
                 x=exp1.stm_vbg(evi_tok11[i][0])
                 if x==None:
                   x=evi_tok11[i][0].lower()
                 evi_tok_new.append(x)
             else:
        
                x=evi_tok11[i][0].lower()
                evi_tok_new.append(x)



          
          evi_filt_sent1=exp1.stop_wd(evi_tok_new)
          evi_lem1=exp1.elim_pun(evi_filt_sent1)
 


          evi_sub1,evi_ver1,evi_obj1,e_sub1,e_ver1,e_obj1=exp1.extract_parts_svo(evi_triples,evi_pos)
          evi_subnmod1,evi_vernmod1,evi_objnmod1,evi_subnmod,evi_vernmod,evi_objnmod=exp1.extract_parts_nmod(evi_triples,e_obj1,e_ver1,e_sub1,evi_pos)
    evi_color_amod,evi_mod=exp1.extract_parts_mod(evi_triples,evi_obj1,evi_pos,evi_subnmod1,evi_vernmod1,colours)
    evi_subcomp,evi_objcomp,evi_sncomp,evi_vncomp,evi_oncomp=exp1.extract_parts_comp(evi_triples,e_obj1,e_ver1,e_sub1,evi_subnmod1,evi_vernmod1,evi_objnmod1,evi_pos)
    evi_subnum,evi_objnum=exp1.extract_parts_num(evi_triples,e_obj1,e_sub1,evi_pos)
    evi_subconj,evi_objconj=exp1.extract_parts_conj(evi_triples,e_obj1,e_sub1,evi_pos)

    for i in evi_subcomp:
        evi_sub1.append(i)
    for i in evi_objcomp:
        evi_obj1.append(i)

    if "'s" in e_ver1:    
       e_ver1.append("be") #is_convert(e_ver1)  

  
    evi_nmod=[]
    if(evi_subnmod1):
       evi_nmod.append(evi_subnmod1)
    if(evi_vernmod1):
        
       evi_nmod.append(evi_vernmod1)
    if(evi_objnmod1):
       evi_nmod.append(evi_objnmod1)
  
    evi_mark,evi_advcl,evi_advmod,evi_neg,evi_prt=exp1.extract_parts_other(evi_triples,evi_pos)

    col=0
    if evi_color_amod:
      col=exp1.colour(hyp_color_amod,evi_color_amod,colours)
    gr=["group","bunch","pair"]
    for i in e_sub1:
      if i in gr :
          if (evi_subnmod1):
             evi_sub1.append(evi_subnmod1[0])
         
            

    syn_sim=exp1.sim_syn(hyp_triples,evi_triples)
    #extracting glove vector n calc dist of sub, ver,n obj...
    
    hyp_sub_gl=exp1.glov_vector(hyp_sub1)


    evi_sub_gl=exp1.glov_vector(evi_sub1)


    #calculating distance.....

    d_sub_max,d_sub_min=exp1.sim(hyp_sub_gl,evi_sub_gl)
    

    hyp_ver_gl=exp1.glov_vector(hyp_ver1)


    evi_ver_gl=exp1.glov_vector(evi_ver1)


    #calculating distance.....

    d_ver_max,d_ver_min=exp1.sim(hyp_ver_gl,evi_ver_gl)
    

    hyp_obj_gl=exp1.glov_vector(hyp_obj1)


    evi_obj_gl=exp1.glov_vector(evi_obj1)


    #calculating distance.....

    d_obj_max,d_obj_min=exp1.sim(hyp_obj_gl,evi_obj_gl)
   
    evi_lem11=[]
    if lab_sample[p]=='entailment':
      for i in evi_lem1:
        if(( i=='wheeled')or(i== 'well')or (i=='else')or(i== "'s") or(i=='body')or(i=='already')or(i=='means') or (i=='follow')or(i=='ed')or(i=='onto')or
           (i=='fur') or (i=='job') or (i=='`') or (i=='area')or(i=="kind") or (i=='yard')or(i=='dr')or(i=='rough')or(i=="might")or (i=='drip')or(i=='least')
           or (i=="onward") or(i=="'") or (i=='along') or (i=='ben')or (i=="set") or  (i=="lot") or (i=="next") or(i=='``')or(i=="''")or (i==""") or (i==""")
           or(i=='something')or(i=='together')or(i=="past")or(i=='somewhere')or(i=='another')or(i=='upon')or(i=="across")):
            continue
        else:
            evi_lem11.append(i)
           
      final_score,match_words,score_ant=exp1.all_words(evi_lem11,hyp_lem1)
    else:
        
      final_score,match_words,score_ant=exp1.all_words(evi_lem1,hyp_lem1)
#
    
    flag_s=exp1.svo_match(hyp_sub1,evi_sub1)
#
    flag_v=exp1.svo_match(hyp_ver1,evi_ver1)
    flag_o=exp1.svo_match(hyp_obj1,evi_obj1)
   
    
    flag_so=exp1.svo_match(hyp_sub1,evi_obj1)
    flag_os=exp1.svo_match(hyp_obj1,evi_sub1)
    

    coup=["man","woman"]

    if(evi_sub1[0]=='couple'):

        if(hyp_sub1[0] in coup):
            if(hyp_subconj):
               if (hyp_subconj[0] in coup):

                   flag_s=1
                   
###########........added code for contradiction detection....############
                   
    #...working/running in kitchen 
    fl_kit=0               
    flag_ver_nmod=exp1.svo_match(hyp_vernmod1,evi_vernmod1)

    if (flag_ver_nmod==1) and ("kitchen" in hyp_vernmod1):
        
        if "work" in hyp_ver1 and "run" in evi_ver1:
            
            fl_kit=1
    flage_ver_nmod=exp1.svo_match(evi_obj1,hyp_vernmod1)
    flagh_ver_nmod=exp1.svo_match(hyp_obj1,evi_vernmod1)

    
    # animal...ppl issue....
    fl_anim=0
    if (flag_s==0) and ("people" in hyp_sub1 ):
        for i in evi_sub1:
            if i in animal:
                fl_anim=1

    #car...bike..
    fl_veh=0            
    four_whl=["car","jeep"]
    two_whl=["bike","scooter"]
    if (flag_o==0):
      for i in evi_obj1:            
         if i in four_whl:
             for j in hyp_obj1:
                 if j in two_whl:
                     fl_veh=1
   #man..lady
    if flag_s==0:
        if ("man" in hyp_sub1) and ("lady" in evi_sub1):
            flag_s=-1

    #....negative detection...
    
    flg_nege=0
    flg_negh=0
    if (evi_neg):
       flg_nege=exp1.neg_detect(evi_neg,hyp_ver1,evi_pos,evi_ver1,hyp_obj1,flag_o)
    if(hyp_neg):
       flg_negh=exp1.neg_detect(hyp_neg,evi_ver1,hyp_pos,hyp_ver1,evi_obj1,flag_o)

    f_neg=0
    if (flg_nege):# or 
         f_neg=1
    elif (flg_negh):
        f_neg=1
    else:
        f_neg=0
        
        #...................number mismatch identification..
        
    f_subnum=exp1.nummod_match(hyp_subnum,evi_subnum)
    f_objnum=exp1.nummod_match(hyp_objnum,evi_objnum)

    if (f_subnum ==1 or f_objnum ==1 ):#or f_objnum1==1):
        f_num=1
    else:
        f_num=0
    #. ..comp:prt negation..
    fl_prt=0    
    for i in evi_prt:    
      x,y,hy,b,hypero,m,ho=exp1.wn_reln(i)
      if y:
        for j in y:  
           if j in hyp_prt:
               fl_prt=1

    thr_c=0.4

    thr_e=0.7
    flag_nmod=0
    ce=0
    cn=0
    ec=0
    en=0
    cc=0
    ee=0
    outc1=0
    oute1=0
    outn1=0
    if((flag_s==flag_v==flag_o==1) or (final_score==1) or (flag_s==flag_v==1 and flag_o==3 and flage_ver_nmod==1)):
         oute1=1
    else:
      if (flag_s==-1)or(flag_v==-1) or(flag_o==-1)or (score_ant==-1) or (fl_anim ==1)or (fl_veh==1) or(fl_prt) or ((flag_v==0) and (d_ver_min<thr_c)) or ((flag_s== 0) and (d_sub_min<thr_c))or((flag_o== 0) and (d_obj_min<thr_c)) or (col==-1) or (f_neg==1) or (f_num==1) or (fl_kit==1)or ((flag_s==flag_v==1) and(flag_ver_nmod==0))or (syn_sim<0.2):
         outc1=1
##     
      else:
          outn1=1
    
#.....writing result to file(CSV)....
     
##    csvData=[[p,lab_sample[p],oute1,outc1,outn1]]
##    with open('exp1_v1_1k.csv', 'a') as csvFile:
##       writer = csv.writer(csvFile)
##       writer.writerows(csvData)
##
##    csvFile.close()           


#exp2.......
##from nltk.stem.wordnet import WordNetLemmatizer
##incor_sent=[]
##for p in range(len(hyp_sample)):
##   if (lab_cor_sample[p]=='yes'):
##
##    hyp_sen1=hyp_sample[p]
##    
##    #------------tokens,stop word ,punctuation removal,lemmatize....
##    lm = WordNetLemmatizer()
##    ps = PorterStemmer()
##    hyp_tok=[]
##    hyp_tok1=[]
##   
##    
##    hyp_sen2=hyp_sen1.split("-")
##    hyp_sen= " " .join(hyp_sen2)
##    hyp_input_sen=hyp_sen
##    hyp_tk=word_tokenize(hyp_input_sen.lower())
##    hyp_tok1=pos_tag(word_tokenize(hyp_input_sen))
##
##    ps=['VBZ','WRB','VBP']
##    pp=['VBD','VBN']
##    nn=['NNS','NNPS']
##    nn_exclude=["building","ceiling","king","ping","ring","singer","something",'spring',"thing","wing"]
    for i in range(len(hyp_tok1)):
      if (hyp_tok1[i][1] in nn):
          x=exp2.stm_nns(hyp_tok1[i][0].lower())
          if x==None:
                  x=hyp_tok1[i][0].lower()

          hyp_tok.append(x)

      elif (hyp_tok1[i][1]=='VBG'):
        x=exp2.stm_vbg(hyp_tok1[i][0].lower())
        if x==None:
                  x=hyp_tok1[i][0].lower()

        hyp_tok.append(x)

      elif (hyp_tok1[i][1] in ps):
        x=exp2.stm_vbz(hyp_tok1[i][0].lower())
        if x==None:
                  x=hyp_tok1[i][0].lower()
        hyp_tok.append(x)

      elif (hyp_tok1[i][1] in pp):
        x=exp2.stm_vbd(hyp_tok1[i][0].lower())
        if x==None:
                  x=hyp_tok1[i][0].lower()
        hyp_tok.append(x)

      elif (hyp_tok1[i][1] =="NN" and re.search("ing",hyp_tok1[i][0]) and (hyp_tok1[i][0] not in nn_exclude)):
            x=exp2.stm_vbg(hyp_tok1[i][0])
            if x==None:
                x=hyp_tok1[i][0].lower()
            hyp_tok.append(x)
      else:
        
        x=hyp_tok1[i][0].lower()
        hyp_tok.append(x)
    hyp_filt_sent=exp2.stop_wd(hyp_tok)
    hyp_lem1=exp2.elim_pun(hyp_filt_sent)

   #---------------------parsing
    
 
    hyp_triples,hyp_pos=exp2.extract_triples(hyp_sen)
    for i in range(len(hyp_triples)):
        if hyp_triples[i][2]=='cop':

            hyp_sub1,hyp_ver1,hyp_obj1,h_sub1,h_ver1,h_obj1=exp2.extract_cop(hyp_triples,hyp_pos)
            hyp_subnmod1,hyp_vernmod1,hyp_objnmod1,hyp_subnmod,hyp_vernmod,hyp_objnmod=exp2.extract_parts_nmod(hyp_triples,hyp_obj1,hyp_ver1,hyp_sub1,hyp_pos)
            break 

        elif hyp_triples[i][2]=='nsubjpass':

            
               hyp_sub1,hyp_ver1,hyp_obj1,h_sub1,h_ver1,h_obj1=exp2.extract_pass(hyp_triples,hyp_pos)
               if (hyp_obj1==[]):

                 hyp_subnmod1,hyp_vernmod1,hyp_objnmod1,hyp_subnmod,hyp_vernmod,hyp_objnmod=exp2.extract_parts_nmod(hyp_triples,h_obj1,h_ver1,h_sub1,hyp_pos)
                 if(hyp_vernmod1): 

                    h_obj1.append(hyp_vernmod1[0])
                    w_pos=exp2.extract_pos(hyp_pos,hyp_vernmod1[0])

                    if w_pos=="NNS" or "NNPS":
                      w=exp2.stm_nns(hyp_vernmod1[0])
                    elif w_pos=="NN":
                        w=hyp_vernmod1[0].lower()

                    if w==None:
                         w=hyp_vernmod1[0].lower()
                    if w not in hyp_obj1:
             
                            hyp_obj1.append(w)
  
               hyp_subnmod1,hyp_vernmod1,hyp_objnmod1,hyp_subnmod,hyp_vernmod,hyp_objnmod=exp2.extract_parts_nmod(hyp_triples,h_obj1,h_ver1,h_sub1,hyp_pos)
                  
                  
               break    

        elif (hyp_triples[i][2]=='nsubj' or hyp_triples[i][2]=='acl'):
                hyp_sub1,hyp_ver1,hyp_obj1,h_sub1,h_ver1,h_obj1=exp2.extract_parts_svo(hyp_triples,hyp_pos)
                hyp_subnmod1,hyp_vernmod1,hyp_objnmod1,hyp_subnmod,hyp_vernmod,hyp_objnmod=exp2.extract_parts_nmod(hyp_triples,h_obj1,h_ver1,h_sub1,hyp_pos)
                break
        else:
            hyp_sub1=[]
            h_sub1=[]
            hyp_ver1=[]
            h_ver1=[]
            hyp_obj1=[]
            h_obj1=[]
            hyp_subnmod1=[]
            hyp_vernmod1=[]
            hyp_objnmod1=[]
            
    if (hyp_sub1==hyp_ver1==hyp_obj1==[]):        

       incor_sent.append(p)
    animal=["dog","cat","horse","cow","elephant"]
    costum=['shirt','costume','dress','skirt','suit','coat','outfit','jersey','top','pants','suits','jacket','trunk','hoodie','sweatshirt','bathrobe','clothing','sandal']
    if((hyp_sub1) and (hyp_subnmod1)):
        for i in hyp_subnmod1:
          if (i in costum):
               hyp_st=exp2.in_correction(h_sub1,hyp_tk)
               hyp_triples,hyp_pos=exp2.extract_triples(hyp_st)

               hyp_tok_new=[]
               hyp_tok11=pos_tag(word_tokenize(hyp_st))
               ps=['VBZ','VBP','WRB']
               pp=['VBD','VBN']
               for i in range(len(hyp_tok11)):
                  if (hyp_tok11[i][1] in nn):
                     x=exp2.stm_nns(hyp_tok11[i][0].lower())
                     if x==None:
                         x=hyp_tok11[i][0].lower()
                     hyp_tok_new.append(x)

                  elif (hyp_tok11[i][1]=='VBG'):
                     x=exp2.stm_vbg(hyp_tok11[i][0].lower())
                     if x==None:
                         x=hyp_tok11[i][0].lower()
                     hyp_tok_new.append(x)

                  elif (hyp_tok11[i][1] in ps):

                     x=exp2.stm_vbz(hyp_tok11[i][0].lower())
                     if x==None:
                           x=hyp_tok11[i][0].lower()
                     hyp_tok_new.append(x)

                  elif (hyp_tok11[i][1] in pp):
                    x=exp2.stm_vbd(hyp_tok11[i][0].lower())
                    if x==None:
                        x=hyp_tok11[i][0].lower()
                    hyp_tok_new.append(x)

                  elif (hyp_tok11[i][1] =="NN" and re.search("ing",hyp_tok11[i][0])and (hyp_tok11[i][0] not in nn_exclude)):
                      x=exp2.stm_vbg(hyp_tok11[i][0])
                      if x==None:
                         x=hyp_tok11[i][0].lower()
                      hyp_tok_new.append(x)
                  else:
        
                     x=hyp_tok11[i][0].lower()
                     hyp_tok_new.append(x)


               
               hyp_filt_sent1=exp2.stop_wd(hyp_tok_new)
               hyp_lem1=exp2.elim_pun(hyp_filt_sent1)
 

    
               hyp_sub1,hyp_ver1,hyp_obj1,h_sub1,h_ver1,h_obj1=exp2.extract_parts_svo(hyp_triples,hyp_pos)


               hyp_subnmod1,hyp_vernmod1,hyp_objnmod1,hyp_subnmod,hyp_vernmod,hyp_objnmod=exp2.extract_parts_nmod(hyp_triples,h_obj1,h_ver1,h_sub1,hyp_pos)

    hyp_color_amod,hyp_mod=exp2.extract_parts_mod(hyp_triples,hyp_obj1,hyp_pos,hyp_subnmod1,hyp_vernmod1)

    hyp_mark,hyp_advcl,hyp_advmod,hyp_neg,hyp_prt=exp2.extract_parts_other(hyp_triples,hyp_pos)

    hyp_subcomp,hyp_objcomp,hyp_sncomp,hyp_vncomp,hyp_oncomp=exp2.extract_parts_comp(hyp_triples,h_obj1,h_ver1,h_sub1,hyp_subnmod1,hyp_vernmod1,hyp_objnmod1,hyp_pos)
    hyp_subnum,hyp_objnum=exp2.extract_parts_num(hyp_triples,h_obj1,h_sub1,hyp_pos)
    hyp_subconj,hyp_objconj=exp2.extract_parts_conj(hyp_triples,h_obj1,h_sub1,hyp_pos)

    for i in hyp_subconj:
        hyp_sub1.append(i)
    for i in hyp_objconj:
        hyp_obj1.append(i)
    
    for i in hyp_subcomp:
        hyp_sub1.append(i)
    for i in hyp_objcomp:
        hyp_obj1.append(i)

    h_ver1=exp2.is_convert(h_ver1)

    gr=["group","bunch","pair"]
    for i in h_sub1:
      if i in gr:
         if(hyp_subnmod1):
            hyp_sub1.append(hyp_subnmod1[0])

    lmtzr = WordNetLemmatizer()
    evi_tok=[]
    evi_sen1=evi_sample[p]
   
    
    evi_sen2=evi_sen1.split("-")
    evi_sen= " " . join(evi_sen2)
    evi_input_sen=evi_sen
    evi_tk=word_tokenize(evi_input_sen.lower())
    evi_tok1=pos_tag(word_tokenize(evi_input_sen))

    ps=['VBZ','VBP','WRB']
    pp=['VBD','VBN']
    nn=['NNS','NNPS']
    for i in range(len(evi_tok1)):
      if (evi_tok1[i][1] in nn):
          x=exp2.stm_nns(evi_tok1[i][0].lower())
          if x==None:
                  x=evi_tok1[i][0].lower()
          evi_tok.append(x)

      elif (evi_tok1[i][1]=='VBG'):
        x=exp2.stm_vbg(evi_tok1[i][0].lower())
        if x==None:
                  x=evi_tok1[i][0].lower()
        evi_tok.append(x)

      elif (evi_tok1[i][1] in ps):
        x=exp2.stm_vbz(evi_tok1[i][0].lower())
        if x==None:
                  x=evi_tok1[i][0].lower()
        evi_tok.append(x)

      elif (evi_tok1[i][1] in pp):
        x=exp2.stm_vbd(evi_tok1[i][0].lower())
        if x==None:
                  x=evi_tok1[i][0].lower()
        evi_tok.append(x)

      elif (evi_tok1[i][1] =="NN" and re.search("ing",evi_tok1[i][0]) and (evi_tok1[i][0] not in nn_exclude)):
         x=exp2.stm_vbg(evi_tok1[i][0])
         if x==None:
            x=evi_tok1[i][0].lower()
         evi_tok.append(x)
      else:
        
        x=evi_tok1[i][0].lower()
        evi_tok.append(x)
    evi_filt_sent=exp2.stop_wd(evi_tok)
    evi_lem1=exp2.elim_pun(evi_filt_sent)

    #---------------------parsing
    evi_triples,evi_pos=exp2.extract_triples(evi_sen)
  
    for i in range(len(evi_triples)):
        if evi_triples[i][2]=='cop':

            evi_sub1,evi_ver1,evi_obj1,e_sub1,e_ver1,e_obj1=exp2.extract_cop(evi_triples,evi_pos)
            evi_subnmod1,evi_vernmod1,evi_objnmod1,evi_subnmod,evi_vernmod,evi_objnmod=exp2.extract_parts_nmod(evi_triples,e_obj1,e_ver1,e_sub1,evi_pos)
            break

        elif evi_triples[i][2]=='nsubjpass':
#.............taking nsubjpass as subj............
               evi_sub1,evi_ver1,evi_obj1,e_sub1,e_ver1,e_obj1=exp2.extract_pass(evi_triples,evi_pos)
               if(evi_obj1==[]):

                 evi_subnmod1,evi_vernmod1,evi_objnmod1,evi_subnmod,evi_vernmod,evi_objnmod=exp2.extract_parts_nmod(evi_triples,e_obj1,e_ver1,e_sub1,evi_pos)
                 if (evi_vernmod1):

                     e_obj1.append(evi_vernmod1[0]) 
                     w_pos=exp2.extract_pos(evi_pos,evi_vernmod1[0])

                     if w_pos=="NNS" or "NNPS":
                       w=exp2.stm_nns(evi_vernmod1[0])
                     elif w_pos=="NN":
                        w=evi_vernmod1[0].lower()

                     if w==None:
                         w=evi_vernmod1[0].lower()
                     if w not in evi_obj1:
             
                            evi_obj1.append(w)

                   
               evi_subnmod1,evi_vernmod1,evi_objnmod1,evi_subnmod,evi_vernmod,evi_objnmod=exp2.extract_parts_nmod(evi_triples,e_obj1,e_ver1,e_sub1,evi_pos)
               break
        elif ((evi_triples[i][2]=='nsubj')or (evi_triples[i][2]=='acl')):   
                 evi_sub1,evi_ver1,evi_obj1,e_sub1,e_ver1,e_obj1=exp2.extract_parts_svo(evi_triples,evi_pos)

                 evi_subnmod1,evi_vernmod1,evi_objnmod1,evi_subnmod,evi_vernmod,evi_objnmod=exp2.extract_parts_nmod(evi_triples,e_obj1,e_ver1,e_sub1,evi_pos)
                 break
        else:
            evi_sub1=[]
            e_sub1=[]
            evi_ver1=[]
            e_ver1=[]
            evi_obj1=[]
            e_obj1=[]
            evi_subnmod1=[]
            evi_vernmod1=[]
            evi_objnmod1=[]
    if (evi_sub1==evi_ver1==evi_obj1==[]):        

       incor_sent.append(p)
     
    costum=['shirt',"short",'costume','dress','skirt','suit','outfit','coat','jersey','top','pants','suits','jacket',
            'trunk','hoodie','sweatshirt','bathrobe','clothing',"vest","tutu"]
    if((evi_sub1) and (evi_subnmod1)):
 
      for i in evi_subnmod1:
        if (i in costum):
        
        
        
        
          evi_st=exp2.in_correction(e_sub1,evi_tk)
          evi_triples,evi_pos=exp2.extract_triples(evi_st)

          evi_tok_new=[]
          evi_tok11=pos_tag(word_tokenize(evi_st))

          ps=['VBZ','VBP','WRB']
          pp=['VBD','VBN']
          for i in range(len(evi_tok11)):
             if (evi_tok11[i][1] in nn):
                x=exp2.stm_nns(evi_tok11[i][0].lower())
                if x==None:
                  x=evi_tok11[i][0].lower()
                evi_tok_new.append(x)

             elif (evi_tok11[i][1]=='VBG'):
                x=exp2.stm_vbg(evi_tok11[i][0])
                if x==None:
                  x=evi_tok11[i][0].lower()
                evi_tok_new.append(x)

             elif (evi_tok11[i][1] in ps):
                x=exp2.stm_vbz(evi_tok11[i][0].lower())
                if x==None:
                  x=evi_tok11[i][0].lower()
                evi_tok_new.append(x)

             elif (evi_tok11[i][1] in pp):
                x=exp2.stm_vbd(evi_tok11[i][0].lower())
                if x==None:
                  x=evi_tok11[i][0].lower()
                evi_tok_new.append(x)

             elif (evi_tok11[i][1] =="NN" and re.search("ing",evi_tok11[i][0])and (evi_tok11[i][0] not in nn_exclude)):
                 x=exp2.stm_vbg(evi_tok11[i][0])
                 if x==None:
                   x=evi_tok11[i][0].lower()
                 evi_tok_new.append(x)
             else:
        
                x=evi_tok11[i][0].lower()
                evi_tok_new.append(x)



          
          evi_filt_sent1=exp2.stop_wd(evi_tok_new)
          evi_lem1=exp2.elim_pun(evi_filt_sent1)



          evi_sub1,evi_ver1,evi_obj1,e_sub1,e_ver1,e_obj1=exp2.extract_parts_svo(evi_triples,evi_pos)
          evi_subnmod1,evi_vernmod1,evi_objnmod1,evi_subnmod,evi_vernmod,evi_objnmod=exp2.extract_parts_nmod(evi_triples,e_obj1,e_ver1,e_sub1,evi_pos)
    evi_color_amod,evi_mod=exp2.extract_parts_mod(evi_triples,evi_obj1,evi_pos,evi_subnmod1,evi_vernmod1)
    evi_subcomp,evi_objcomp,evi_sncomp,evi_vncomp,evi_oncomp=exp2.extract_parts_comp(evi_triples,e_obj1,e_ver1,e_sub1,evi_subnmod1,evi_vernmod1,evi_objnmod1,evi_pos)
    evi_subnum,evi_objnum=exp2.extract_parts_num(evi_triples,e_obj1,e_sub1,evi_pos)
    evi_subconj,evi_objconj=exp2.extract_parts_conj(evi_triples,e_obj1,e_sub1,evi_pos)

    for i in evi_subcomp:
        evi_sub1.append(i)
    for i in evi_objcomp:
        evi_obj1.append(i)
  
    if "'s" in e_ver1:    
       e_ver1.append("be") #is_convert(e_ver1)  
    evi_nmod=[]
    if(evi_subnmod1):
       evi_nmod.append(evi_subnmod1)
    if(evi_vernmod1):
        
       evi_nmod.append(evi_vernmod1)
    if(evi_objnmod1):
       evi_nmod.append(evi_objnmod1)
   
    evi_mark,evi_advcl,evi_advmod,evi_neg,evi_prt=exp2.extract_parts_other(evi_triples,evi_pos)

    col=0
    if evi_color_amod:
      col=exp2.colour(hyp_color_amod,evi_color_amod,colours)
    if evi_mod:
        
      col=exp2.colour(hyp_mod,evi_mod,colours)
      
    gr=["group","bunch","pair"]
    for i in e_sub1:
      if i in gr :
          if (evi_subnmod1):
             evi_sub1.append(evi_subnmod1[0])
         
            

    match,c=exp2.triple_match(hyp_triples,evi_triples)

    syn_sim=exp2.sim_syn(hyp_triples,evi_triples)
    #extracting glove vector n calc dist of sub, ver,n obj...
    
    hyp_sub_gl=exp2.glov_vector(hyp_sub1)


    evi_sub_gl=exp2.glov_vector(evi_sub1)


    #calculating distance.....

    d_sub_max,d_sub_min=exp2.sim(hyp_sub_gl,evi_sub_gl)
    

    hyp_ver_gl=exp2.glov_vector(hyp_ver1)


    evi_ver_gl=exp2.glov_vector(evi_ver1)


    #calculating distance.....

    d_ver_max,d_ver_min=exp2.sim(hyp_ver_gl,evi_ver_gl)
    

    hyp_obj_gl=exp2.glov_vector(hyp_obj1)


    evi_obj_gl=exp2.glov_vector(evi_obj1)


    #calculating distance.....

    d_obj_max,d_obj_min=exp2.sim(hyp_obj_gl,evi_obj_gl)
    

    evi_lem11=[]
    if lab_sample[p]=='entailment':
      for i in evi_lem1:
        if(( i=='wheeled')or(i== 'well')or (i=='else')or(i== "'s") or(i=='body')or(i=='already')or(i=='means') or (i=='follow')or(i=='ed')or(i=='onto')or
           (i=='fur') or (i=='job') or (i=='`') or (i=='area')or(i=="kind") or (i=='yard')or(i=='dr')or(i=='rough')or(i=="might")or (i=='drip')or(i=='least')
           or (i=="onward") or(i=="'") or (i=='along') or (i=='ben')or (i=="set") or  (i=="lot") or (i=="next") or(i=='``')or(i=="''")or (i==""") or (i==""")
           or(i=='something')or(i=='together')or(i=="past")or(i=='somewhere')or(i=='another')or(i=='upon')or(i=="across")):
            continue
        else:
            evi_lem11.append(i)
           
      final_score,match_words,score_ant,score_ant1=exp2.all_words(evi_lem11,hyp_lem1)
    else:
        
      final_score,match_words,score_ant,score_ant1=exp2.all_words(evi_lem1,hyp_lem1)

    

    
    flag_s=exp2.svo_match(hyp_sub1,evi_sub1)

    flag_v=exp2.svo_match(hyp_ver1,evi_ver1)
    flag_o=exp2.svo_match(hyp_obj1,evi_obj1)
   #...cross check...sub n obj...
    
    flag_so=exp2.svo_match(hyp_sub1,evi_obj1)
    flag_os=exp2.svo_match(hyp_obj1,evi_sub1)
    

    coup=["man","woman"]

    if(evi_sub1[0]=='couple'):

        if(hyp_sub1[0] in coup):
            if(hyp_subconj):
               if (hyp_subconj[0] in coup):

                   flag_s=1
                   
###########........added code for contradiction detection....############
                   
    #...working/running in kitchen 
    fl_kit=0               
    flag_ver_nmod=exp2.svo_match(hyp_vernmod1,evi_vernmod1)
##    # print("fl_kit",fl_kit)
    if (flag_ver_nmod==1) and ("kitchen" in hyp_vernmod1):
        
        if "work" in hyp_ver1 and "run" in evi_ver1:
            
            fl_kit=1
    flage_ver_nmod=exp2.svo_match(evi_obj1,hyp_vernmod1)
    flagh_ver_nmod=exp2.svo_match(hyp_obj1,evi_vernmod1)
    fl_anim=0
    if (flag_s==0) and ("people" in hyp_sub1 ):
        for i in evi_sub1:
            if i in animal:
                fl_anim=1

    #car...bike..
    fl_veh=0            
    four_whl=["car","jeep"]
    two_whl=["bike","scooter"]
    if (flag_o==0):
      for i in evi_obj1:            
         if i in four_whl:
             for j in hyp_obj1:
                 if j in two_whl:
                     fl_veh=1
   #man..lady
    if flag_s==0:
        if ("man" in hyp_sub1) and ("lady" in evi_sub1):
            flag_s=-1

    #....negative detection...
    
    flg_nege=0
    flg_negh=0
    if (evi_neg):
       flg_nege=exp2.neg_detect(evi_neg,hyp_ver1,evi_pos,evi_ver1,hyp_obj1,flag_o)
    if(hyp_neg):
       flg_negh=exp2.neg_detect(hyp_neg,evi_ver1,hyp_pos,hyp_ver1,evi_obj1,flag_o)

    f_neg=0
    if (flg_nege):# or 
         f_neg=1
    elif (flg_negh):
        f_neg=1
    else:
        f_neg=0
        
        #...................number mismatch identification..
        
    f_subnum=exp2.nummod_match(hyp_subnum,evi_subnum)
    f_objnum=exp2.nummod_match(hyp_objnum,evi_objnum)

    
    f_objnum1=exp2.nummod_match(hyp_objnmod,evi_objnmod)

    if (f_subnum ==1 or f_objnum ==1 or f_objnum1==1):
        f_num=1
    else:
        f_num=0
    #. ..comp:prt negation..
    fl_prt=0    
    for i in evi_prt:    
      x,y,hy,b,hypero,m,ho=exp2.wn_reln(i)
      if y:
        for j in y:  
           if j in hyp_prt:
               fl_prt=1

    thr_c=0.4

    thr_e=0.7
    flag_nmod=0
    outc2=0
    outn2=0
    oute2=0
    
    if((flag_s==flag_v==flag_o==1) or (final_score==1) or (flag_s==flag_v==1 and flag_o==3 and flage_ver_nmod==1) ):#or (flag_s==flag_v==1 and flag_o==2 and flag_ver_nmod==1)): 
        oute2=1
    
    if (flag_s==-1)or(flag_v==-1) or(flag_o==-1)  or (score_ant==-1) or  (score_ant1==-1)or (fl_anim ==1)or (fl_veh==1) or(fl_prt) or ((flag_v==0) and (d_ver_min<thr_c)) or ((flag_s== 0) and (d_sub_min<thr_c))or((flag_o== 0) and (d_obj_min<thr_c)) or (col==-1) or (f_neg==1) or (f_num==1) or (fl_kit==1)or ((flag_s==flag_v==1) and(flag_ver_nmod==0))or (syn_sim<0.2):
         outc2=1
    else :    
        outn2=1  




#sent classification..
    
    if (oute1==1):
      claslab='E'
      #counte+=1
    elif(outc2==1)and (outn1==0):
      claslab='C'
      #countc+=1
    elif((outn1==1)or(outn2==1)):
      claslab='N'
      #count+=1
#counting total no in each classes in the dataset..
        
    if lab_sample[p]=="entailment":
       cnt_e+=1
    elif lab_sample[p]=="contradiction":
        cnt_c+=1
    else:
        cnt_n+=1
    cccc=0 
    if ((lab_sample[p]=="entailment") and (claslab=='E')):
        counte+=1
    elif ((lab_sample[p]=="contradiction") and (claslab=='C')):
        countc+=1
    elif ((lab_sample[p]=="neutral") and (claslab=='N')):
        countn+=1
    else:
        cccc+=1
        
        
       
#.....writing result to file(CSV)....
    
    csvData=[[p,lab_sample[p],oute1,oute2,outc1,outc2,outn1,outn2,claslab]]
    with open('sof_imp_trial.csv', 'a') as csvFile:
       writer = csv.writer(csvFile)
       writer.writerows(csvData)

    csvFile.close()           
      
acc=round((((counte+countc+countn)/(p+1))*100),2)
print("total accuracy\n",acc)
acc_e=round(((counte/cnt_e)*100),2)
print("Entailment accuracy\n",acc_e)
acc_c=round(((countc/cnt_c)*100),2)
print("Contradiction accuracy\n",acc_c)
acc_n=round(((countn/cnt_n)*100),2)
print("Neutral accuracy\n",acc_n)


#.....writing result(accracy) to file(CSV)....
header=['Total_Accuracy','Entail_Accuracy','Contrad_Accuracy','Neutral_Accuracy']    
csvData=[[acc,acc_e,acc_c,acc_n]]
with open('sof_imp_trial1.csv', 'a') as csvFile:
    
       writer = csv.writer(csvFile)
       writer.writerow(header)
       writer.writerows(csvData)

csvFile.close()           
