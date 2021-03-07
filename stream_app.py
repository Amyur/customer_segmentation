import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("""
# Segmentación de clientes usando Unsupervised Learning
El conjunto de datos hace referencia al gasto realizado a lo largo de un año por los clientes de un distribuidor mayorista.
El gasto se encuentra en unidades monetarias (m.u.) para diversas categorías de productos, como lo son:
refrescos, lacteos, comestibles, congelados, alimentos especializados, detergentes y productos de papel.
""")

if st.button("Ver exploración de los datos"):
    df_3 = pd.read_csv("data.csv", sep=",")
    fig1, ax = plt.subplots()
    ax = df_3['Channel'].value_counts().plot(kind='bar',
                                    figsize=(5,3),
                                    title="Frecuencia por canal")
    ax.set_xlabel("Canal")
    ax.set_ylabel("Frecuencia")
    st.write(fig1)

    fig2, ax = plt.subplots()
    ax = df_3['Region'].value_counts().plot(kind='bar',
                                    figsize=(5,3),
                                    title="Frecuencia por región")
    ax.set_xlabel("Región")
    ax.set_ylabel("Frecuencia")
    st.pyplot(fig2)

    fig3, ax = plt.subplots()
    ax = df_3.groupby(["Channel"]).mean().plot(kind='bar',
                                    figsize=(8,5),
                                    title="Promedio de gasto anual por canal")
    ax.set_xlabel("Canal")
    ax.set_ylabel("Gasto promedio anual")
    st.pyplot()

    fig4, ax = plt.subplots()
    ax = df_3.groupby(["Region"]).mean().plot(kind='bar',
                                    figsize=(8,5),
                                    title="Promedio de gasto anual por región")
    ax.set_xlabel("Región")
    ax.set_ylabel("Gasto promedio anual")
    ax.legend(loc=(0.9,0.6))
    st.pyplot()

    fig5, axs = plt.subplots(6,2,figsize=(15,24))
    fig5.suptitle('Distribución de las variables cuantitativas', fontsize=30)
    sns.distplot(df_3["Fresh"], ax=axs[0, 0])
    sns.boxplot(df_3["Fresh"], ax=axs[0, 1])
    sns.distplot(df_3["Milk"], ax=axs[1, 0])
    sns.boxplot(df_3["Milk"], ax=axs[1, 1])
    sns.distplot(df_3["Grocery"], ax=axs[2, 0])
    sns.boxplot(df_3["Grocery"], ax=axs[2, 1])
    sns.distplot(df_3["Frozen"], ax=axs[3, 0])
    sns.boxplot(df_3["Frozen"], ax=axs[3, 1])
    sns.distplot(df_3["Detergents_Paper"], ax=axs[4, 0])
    sns.boxplot(df_3["Detergents_Paper"], ax=axs[4, 1])
    sns.distplot(df_3["Delicassen"], ax=axs[5, 0])
    sns.boxplot(df_3["Delicassen"], ax=axs[5, 1])
    st.pyplot(fig5)

    fig6, axs = plt.subplots(1, 1)
    sns.set(style="ticks", color_codes=True)
    sns.pairplot(df_3.loc[:, ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]],
                kind="reg", plot_kws={'line_kws':{'color':'red'}}, corner=True)
    plt.suptitle("Relación entre las variables cuantitativas", size=30)
    st.pyplot()


st.write("""
El modelo usado para hallar clusters de clientes fue **K-means** usando la API **Scikit-learn**, 
los datos utilizados se encuentran en este [sitio](https://archive.ics.uci.edu/ml/datasets/Wholesale+customers).
""")

st.sidebar.header('Parámetros de entrada')
st.sidebar.write("(en unidades monetarias m.u)")



def input_features():
        fresh = st.sidebar.slider('Frescos', 3, 60000, 49732)
        milk = st.sidebar.slider('Lacteos', 55, 40000, 3674)
        grocery = st.sidebar.slider('Comestible', 3, 40000, 9130)
        frozen = st.sidebar.slider('Congelados', 25, 20000, 18525)
        detergents_paper = st.sidebar.slider('Detergentes y productos de papel', 3, 10000, 2881)
        delicassen = st.sidebar.slider('Alimentos especializados', 3, 10000, 1524)
        region = st.sidebar.radio("Región",(1, 2, 3))
        channel = st.sidebar.radio("Canal de distribución",(1, 2))

        data = {'Channel':channel,
                'Region':region,
                'Fresh': fresh,
                'Milk': milk,
                'Grocery': grocery,
                'Frozen': frozen,
                'Detergents_Paper': detergents_paper,
                'Delicassen': delicassen
                }
        features = pd.DataFrame(data, index=[0])
        return(features)

df = input_features()

def load_data(df):
    df_1= pd.read_csv("data.csv", sep=",")
    df_2 = df_1
    df_2 = df_2.append(df)
    df_1 = df_1.append(df)
    dummies = df_1[["Channel", "Region"]]
    df_1 = df_1.drop(["Channel", "Region"], axis=1)
    dummies_gen = pd.get_dummies(dummies, columns=["Channel", "Region"])
    df_1 = pd.concat([df_1, dummies_gen], axis=1)

    sc = StandardScaler()
    data_scaled = sc.fit_transform(df_1)
    data_scaled_2 = sc.fit_transform(df_2)
    pca = PCA(n_components=3)
    pca_fitted = pca.fit_transform(data_scaled)
    principalDf = pd.DataFrame(data = pca_fitted[:,:], columns=["pc1","pc2","pc3"])

    clustering_kmeans = KMeans(n_clusters=2, precompute_distances="auto", n_jobs=-1,  random_state=3425)
    principalDf["clusters"] = clustering_kmeans.fit_predict(data_scaled_2)
    df_2["clusters"] = principalDf["clusters"]

    prediction = clustering_kmeans.predict(df)

    principalDf.iloc[-1, 3] = 2

    return(principalDf, prediction, df_2)

principalDf, prediction, df_2 = load_data(df)

st.write("En total se hallaron 2 clusters, a continuación se presenta un breve resumen de cada cluster: ")
cluster_0 = df_2[df_2["clusters"]==0]
st.write("#### Cluster 0")
st.write(cluster_0.loc[:,["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]].describe())
cluster_1 = df_2[df_2["clusters"]==1]
st.write("#### Cluster 1")
st.write(cluster_1.loc[:,["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]].describe())

st.write("""
### **Usted mismo puede modificar los parámetros de entrada en la barra derecha, y de este modo identificar el cluster al que pertenece.** 
""")

st.write('**Parámetros ingresados**')
st.write(df)

st.subheader('Cluster al que pertenece: ')
st.write("### **Cluster {}**".format(int(prediction)))


fig = go.Figure(data=[go.Scatter3d(x=principalDf["pc3"], 
                                        y=principalDf["pc1"], 
                                        z=principalDf["pc2"],
                                        mode='markers',  
                                        marker=dict(size=6,
                                                    color=principalDf["clusters"],                
                                                    colorscale='Viridis',   
                                                    opacity=0.9))])

fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
st.plotly_chart(fig)

st.write("***Sientase libre de inteactuar con el gráfico***")
