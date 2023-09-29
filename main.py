from fastapi import FastAPI
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise        import cosine_similarity
from sklearn.metrics.pairwise        import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer


app=FastAPI(debug=True)

df = pd.read_csv('df_final.csv')


@app.get('/')
def message():
    return 'PROYECTO INTEGRADOR ALEXIS ALVAREZ'

@app.get('/PlayTimeGenre/')
def PlayTimeGenre(genre: str) -> dict:
    genre = genre.capitalize()
    genre_df = df[df[genre] == 1]
    year_playtime_df = genre_df.groupby('year')['playtime_forever'].sum().reset_index()
    max_playtime_year = year_playtime_df.loc[year_playtime_df['playtime_forever'].idxmax(), 'year']
    return {"Género": genre, "Año de lanzamiento con más horas jugadas para Género :": int(max_playtime_year)}

@app.get('/UserForGenre/')
def UserForGenre(genre: str) -> dict:
    genre = genre.capitalize()
    genre_df = df[df[genre] == 1]
    max_playtime_user = genre_df.loc[genre_df['playtime_forever'].idxmax(), 'user_id']
    year_playtime_df = genre_df.groupby('year')['playtime_forever'].sum().reset_index()
    playtime_list = year_playtime_df.to_dict(orient='records')
    result = {
        "Usuario con más horas jugadas para Género " + genre: max_playtime_user,
        "Horas jugadas": playtime_list}
    return result


@app.get('/UsersRecommend/')
def UsersRecommend(year: int) -> dict:
    df_filtrado = df[(df['year'] == year) & (df['recommend'] == True) & (df['sentiment_score'] == 2)]
    if df_filtrado.empty:
        return {"error": 'Valor no encontrado'}
    df_ordenado = df_filtrado.sort_values(by='sentiment_score', ascending=False)
    top_3_reseñas = df_ordenado.head(3)
    resultado = {
        "Puesto 1": top_3_reseñas.iloc[0]['title'],
        "Puesto 2": top_3_reseñas.iloc[1]['title'],
        "Puesto 3": top_3_reseñas.iloc[2]['title']
    }
    return resultado

@app.get('/UsersNotRecommed/')
def UsersRecommend(year: int) -> dict:
    df_filtrado = df[(df['year'] == year) & (df['recommend'] == False) & (df['sentiment_score'] <= 1)]
    if df_filtrado.empty:
        return {"error": 'Valor no encontrado'}
    df_ordenado = df_filtrado.sort_values(by='sentiment_score', ascending=False)
    top_3_reseñas = df_ordenado.head(3)
    resultado = {
        "Puesto 1": top_3_reseñas.iloc[0]['title'],
        "Puesto 2": top_3_reseñas.iloc[1]['title'],
        "Puesto 3": top_3_reseñas.iloc[2]['title']
    }
    return resultado

@app.get('/sentiment_analysis/')
def sentiment_analysis(year: int) -> dict:
    filtered_df = df[df['year'] == year]
    sentiment_counts = filtered_df['sentiment_score'].value_counts()
    result = {
        "Positive": int(sentiment_counts.get(0, 0)),
        "Neutral": int(sentiment_counts.get(1, 0)),
        "Negative": int(sentiment_counts.get(2, 0))
    }
    return result



muestra = df.head(4000)
tfidf = TfidfVectorizer(stop_words='english')
muestra=muestra.fillna("")

tdfid_matrix = tfidf.fit_transform(muestra['review'])
cosine_similarity = linear_kernel( tdfid_matrix, tdfid_matrix)

@app.get('/recomendacion_id/{id_producto}')
def recomendacion(id_producto: int):
    if id_producto not in muestra['steam_id'].values:
        return {'mensaje': 'No existe el id del juego.'}
    
    # Obtener géneros del juego con el id_producto
    generos = muestra.columns[2:17]  # Obtener los nombres de las columnas de género
    
    # Filtrar el dataframe para incluir juegos con géneros coincidentes pero con títulos diferentes
    filtered_df = muestra[(muestra[generos] == 1).any(axis=1) & (muestra['steam_id'] != id_producto)]
    
    # Calcular similitud del coseno
    tdfid_matrix_filtered = tfidf.transform(filtered_df['review'])
    cosine_similarity_filtered = linear_kernel(tdfid_matrix_filtered, tdfid_matrix_filtered)
    
    idx = muestra[muestra['steam_id'] == id_producto].index[0]
    sim_cosine = list(enumerate(cosine_similarity_filtered[idx]))
    sim_scores = sorted(sim_cosine, key=lambda x: x[1], reverse=True)
    sim_ind = [i for i, _ in sim_scores[1:6]]
    sim_juegos = filtered_df['title'].iloc[sim_ind].values.tolist()
    
    return {'juegos recomendados': list(sim_juegos)}

@app.get('/recomendacion_juego/{id_juego}')
def recomendacion_juego(id_juego: int):
    if id_juego not in muestra['id'].values:
        return {'mensaje': 'No existe el id del juego.'}
    titulo = muestra.loc[muestra['id'] == id_juego, 'title'].iloc[0]
    idx = muestra[muestra['title'] == titulo].index[0]
    sim_cosine = list(enumerate(cosine_similarity[idx]))
    sim_scores = sorted(sim_cosine, key=lambda x: x[1], reverse=True)
    sim_ind = [i for i, _ in sim_scores[1:6]]
    sim_juegos = muestra['title'].iloc[sim_ind].values.tolist()
    return {'juegos recomendados': list(sim_juegos)}