# PI_ML_OPS
<p align=center><img src=https://d31uz8lwfmyn8g.cloudfront.net/Assets/logo-henry-white-lg.png><p>

# <h1 align=center> **PROYECTO INDIVIDUAL Nº1** </h1>

# <h1 align=center>**`Machine Learning Operations (MLOps)`**</h1>

<p align=center><img src=https://www.folio3.ai/wp-content/uploads/2023/03/Asset-4-788x301.png><p>

## **Introducción:**
Este proyecto consiste en crear una API que utiliza un modelo de recomendación para Steam, una plataforma multinacional de videojuegos, basado en Machine Learning. El objetivo es crear un sistema de recomendación de videojuegos para usuarios. La API ofrece una interfaz intuitiva para que los usuarios puedan obtener informacion para el sistema de recomendacion y datos sobre generos o fechas puntuales. 

## **Herramientas Utilizadas**
+ Pandas
+ Matplotlib
+ Numpy
+ Seaborn
+ Wordcloud
+ NLTK
+ Uvicorn
+ Render
+ FastAPI
+ Python
+ Scikit-Learn

## **Paso a paso:**
### 1. ETL
Realizamos un proceso de ETL (Extracción, Transformación y Carga) en el que extrajimos datos de diferentes fuentes, los transformamos según las necesidades del proyecto y los cargamos en un destino final para su análisis y uso posterior. Las herramientas primordiales utilizadas fueron python, pandas, sklear y FastApi
### 2. Deployment de la API
Creamos una API utilizando el módulo FastAPI de Python, creando 5 funciones para que puedan ser consultadas:
- def PlayTimeGenre( genero : str ): Debe devolver año con mas horas jugadas para dicho género.
  Ejemplo de input: casual
- def UserForGenre( genero : str ): Debe devolver el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año.
  Ejemplo de input: action
- def UsersRecommend( año : int ): Devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado. (reviews.recommend = True y comentarios positivos/neutrales)
  Ejemplo de input: 2012
- def UsersNotRecommend( año : int ): Devuelve el top 3 de juegos MENOS recomendados por usuarios para el año dado. (reviews.recommend = False y comentarios negativos)
  Ejemplo de input: 2009
- def sentiment_analysis( año : int ): Según el año de lanzamiento, se devuelve una lista con la cantidad de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento.
  Ejemplo de input: 2014
  
Luego realizamos el deployement de esta API utilizando Render. 
Las herramientas utilizadas fueron: Uvicorn, Render, FastAPI
### 3. EDA
Realizamos un proceso de EDA (Exploratory Data Analysis) en el que exploramos y analizamos los datos de manera exhaustiva con el objetivo de obtener insights, identificar patrones, tendencias y relaciones, y así tomar decisiones fundamentadas en base a la información obtenida. Intentando asi obtener alguna pista para crear nuestro modelo de ML
Las herramientas utilizadas fueron: Numpy, Pandas, Matplotlib, Seaborn, Wordcloud, NLTK
### 4. Modelo de Machine Learning
Realizamos un modelo de Machine Learning para generar recomendaciones juegoss, utilizando algoritmos y técnicas como la similitud del coseno y scikit-lear, con el fin de brindar recomendaciones personalizadas y precisas basadas en los gustos y preferencias de cada usuario.
 Si es un sistema de recomendación item-item:

- def recomendacion_juego( id de producto ): Ingresando el id de producto, deberíamos recibir una lista con 5 juegos recomendados similares al ingresado.

 Ejemplo de uso: 70
 Si es un sistema de recomendación user-item:

- def recomendacion_usuario( id de usuario ): Ingresando el id de un usuario, deberíamos recibir una lista con 5 juegos recomendados para dicho usuario.
 Ejemplo de uso: 76561198030567998


La herramienta utilizada fue: Scikit-Learn con las librerias: TfidfVectorizer, linear_kernel, cosine_similarity
Tambien son consultables en la API

## **Links:**
- [Deploy de la API en Render](https://ml-ops-alex.onrender.com/)
- 
## **Canales de contacto:**
+ Linkedin: [Alexis Alvarez](https://www.linkedin.com/in/alvarezalexiscv/)
+ Mail : Alexisalvarezcv@outlook.com
