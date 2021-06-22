############################################
# RATING PRODUCTION SORTING REVIEWS
############################################

############################################
#İş Problemi
############################################
# Amazon'un Elektronik kategorisinde en fazla satış yapan bir ürünün ratingini güncel yorumlara göre hesaplamak ve ürün sayfasında görünmesini istediğimiz en faydalı 20 yorumu seçmek.

###Değişkenler###
# reviewerID – Kullanıcı ID’si       #Örn: A2SUAM1J3GNN3B
# asin – Ürün ID’si.                 #Örn: 0000013714
# reviewerName – Kullanıcı Adı
# helpful – Faydalı yorum derecesi   # [1,2] : toplamda 2 oy almış, bunlerdan 1 tanesi helpful_yes
# reviewText – Yorum                 # Kullanıcının yazdığı inceleme metni
# overall – Ürün rating’i            #1-5 arasında
# summary – İnceleme özeti
# unixReviewTime – İnceleme zamanı   #Unix Zaman, 1 Ocak 1970 (01/01/1970) den beridir geçen saniye sayısına denilen sayısal veri tipidir. (reviewtime değişkeni çevrilmiş halidir)
# reviewTime – İnceleme zamanı
# day_diff  -yorumun yapıldığı tarihten itibaren geçen gün sayısı
# helpful_yes -yorumların faydalı olup olmadığı
# total_vote : toplam aldığı oy


import pandas as pd
import math
import scipy.stats as st
pd.set_option('display.max_columns', None) #tüm sütunları gösterir
pd.set_option('display.max_rows', None) #tüm sstırları gösterir
pd.set_option('display.expand_frame_repr', False) #çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.float_format', lambda x: '%.5f' % x)  #ondalıklı sayıların virgülden sonra 5 hanesini göster
from helper.helper import check_df
df= pd.read_csv("hafta5/ödevler/amazon_review.csv", low_memory=False)

cols=["unixReviewTime","helpful"]       #verimizde aynı bilgiyi gösteren bazı kolonları çıkaralım
all_cols = [i for i in df.columns if i not in cols]
df=df[all_cols]
check_df(df)
df[df["total_vote"]==0].shape  #oylanmayan yorum sayımız
df[df["total_vote"]!=0].shape #oylanan yorum sayımız

df["asin"].nunique()  #tek ürün için analiz yapacağız, kontrol ediyoruz  #1

############################################
#Görev 1:Average Rating’i güncel yorumlara göre  hesaplayınız ve var olan average rating ile kıyaslayınız.
############################################

df["overall"].mean()   #rating ortalaması: 4.58758

#verideki date_diff değişkeninden gidilebilir ama  veride böyle bir bilgi olmayabilir, aşağıdaki gibi hesaplanmaktadır
df['reviewTime'] = pd.to_datetime(df['reviewTime'])  #reviewtime tarih formatına çeviriyoruz
df["reviewTime"].max()  #en yakın hangi tarihte yorum yapılmış
currentdate=current_date = pd.to_datetime('2014-12-08 0:0:0')  #en yakın tarih+1 gün alıyorum (aslında day_diff kolonuyla aynı )
df["days"] = (current_date - df['reviewTime']).dt.days        #yapılan yorumların üzerindenkaç gün geçmiş
df.head()

# geçen süreyi 4 aralığa bölüyoruz ve bu aralıkları etiketliyoruz
df['day_cut']=pd.qcut(df["days"],4,labels=["new","medium","old","very old"])
df.head(20)


#etiketlediğim her aralığa bir ağırlık belirliyorum new:28, medium:26, old:24, very_old:22 ve ağırlıklı ort. alıyorum
def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[dataframe["day_cut"]=="new", "overall"].mean() * w1 / 100 + \
           dataframe.loc[dataframe["day_cut"]=="medium","overall"].mean() * w2 / 100 + \
           dataframe.loc[dataframe["day_cut"]=="old", "overall"].mean() * w3 / 100 + \
           dataframe.loc[dataframe["day_cut"]=="very old", "overall"].mean() * w4 / 100

time_based_weighted_average(df)   # 4.59559
df["overall"].mean()   #rating ortalaması: 4.58758
# overall rating ortalaması ile hesapladığımız ağırlıklı orlama arasonda 0.08 lik bir fark çıkıyor

# bu ürünün new segmentindeki yorumlarına daha fazla ağırlık verildiği zaman ortalmanın daha da arttığını görüyoruz
time_based_weighted_average(df,w1=35,w2=30,w3=20,w4=15)  #4.61564

#ürün puanı eskiden yeniye artarken, yapılan yorumların faydalı bulunması azalmış görülüyor,
df.groupby("day_cut").agg({"overall":"mean","reviewText":"count","total_vote":"sum","helpful_yes":"mean"})

df[df["total_vote"]>100]
#ürünün yorumlarının oy oranı düşük, bunu arttırmak için bir takım çalışmalar yapılabilir.
#Ürün yorumlarının oylanabilmesi için okunması önemli, bazı yorumlar çok uzun olduğu için okunmuyor-kelime sınırı getirilebilir
#yorumların oylama tuşları renkli yapılabilir.

############################################
#Görev 2
############################################
#Ürün için ürün detay sayfasında görüntülenecek 20 review’i belirleyiniz.
#ürüne yapılan yorumlar helpful mu değilmi buna bakara sıralayacağız

#df["total_vote"] :kaç oy almışım df["helpful_yes"]:bu oyların kaçı "yes" yani yorum beğenilmiş
df["helpful_no"]=df["total_vote"]-df["helpful_yes"]
df.head(20)

###################yes-n farkı ve yes/no oranlarına göre bakalım#######################################################
# def yes_no_difference(dataframe,col1,col2):   #calısmadı bakılacak
#     dataframe["yes_no_diff"]=df[col1]-df[col2]
#     return dataframe["yes_no_diff"]
# yes_no_difference(df,"helpful_yes","helpful_no")
# df.sort_values("yes_no_diff",ascending=False).head(20)
#
# def yes_no_average(pos, neg):
#     if pos + neg == 0:
#         return 0
#     else:
#         return pos / (pos + neg)
# df["yes_no_average"]=df.apply(lambda x: yes_no_average(x["helpful_yes"],x["helpful_no"]),axis=1)
#burada uyguladığımız yöntemler bizlere doğru bir degerlendirme olmayacaktır, dah abilimsel bir yöntem kullanmak gerekmektedir##
################################################################################################################################

df.head()
def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df['wilson_lower_bound'] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)  #verimize wilson fonk. uyguluyorz
top_20_review=df.sort_values("wilson_lower_bound",ascending=False).head(20)  #wilson_lower methoduna göre ilk 20 sıralayarak, farklı bir dataframe olarak kaydediyoruz


