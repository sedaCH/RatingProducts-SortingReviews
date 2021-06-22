############################################
# RATING PRODUCTION SORTING REVIEWS
############################################

import pandas as pd
import math
import scipy.stats as st
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
from helper.helper import check_df
df= pd.read_csv("hafta5/ödevler/reviews.csv", low_memory=False)

cols=["unixReviewTime","helpful"]
all_cols = [i for i in df.columns if i not in cols]
df=df[all_cols]
check_df(df)
df[df["total_vote"]==0].shape
df[df["total_vote"]!=0].shape
df["asin"].nunique()

############################################
Calculate the Average Rating according to the current comments and compare it with the existing average rating.
############################################

df["overall"].mean()

#
df['reviewTime'] = pd.to_datetime(df['reviewTime'])
df["reviewTime"].max()
currentdate=current_date = pd.to_datetime('2014-12-08 0:0:0')
df["days"] = (current_date - df['reviewTime']).dt.days
df.head()


df['day_cut']=pd.qcut(df["days"],4,labels=["new","medium","old","very old"])
df.head(20)


def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[dataframe["day_cut"]=="new", "overall"].mean() * w1 / 100 + \
           dataframe.loc[dataframe["day_cut"]=="medium","overall"].mean() * w2 / 100 + \
           dataframe.loc[dataframe["day_cut"]=="old", "overall"].mean() * w3 / 100 + \
           dataframe.loc[dataframe["day_cut"]=="very old", "overall"].mean() * w4 / 100

time_based_weighted_average(df)


# bu ürünün new segmentindeki yorumlarına daha fazla ağırlık verildiği zaman ortalmanın daha da arttığını görüyoruz
time_based_weighted_average(df,w1=35,w2=30,w3=20,w4=15)  #4.61564


df.groupby("day_cut").agg({"overall":"mean","reviewText":"count","total_vote":"sum","helpful_yes":"mean"})


############################################
# Set the 20 reviews to be displayed on the product detail page for the product.
############################################

df["helpful_no"]=df["total_vote"]-df["helpful_yes"]
df.head(20)


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

df['wilson_lower_bound'] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)
top_20_review=df.sort_values("wilson_lower_bound",ascending=False).head(20)


