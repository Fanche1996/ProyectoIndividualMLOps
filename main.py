from ast import literal_eval
from fastapi import FastAPI
from typing import List, Union
from collections import Counter
import pandas as pd
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import numpy as np
import json

model = load("linear_model.pkl")

app = FastAPI()

data = pd.read_csv("steam_games.csv", keep_default_na=False).to_dict(orient="records")

 
data = [entry for entry in data if isinstance(entry['release_date'], str) and entry['release_date'] not in ['NaN', 'soon..']]


def safe_eval(s: str):
    try:
        return eval(s)
    except SyntaxError:
        return []

@app.get("/genero/")
def genero(year: str) -> dict:
    
    # Filtrado por el año
    filtered_data = [game['genres'] for game in data if game['release_date'].startswith(year)]

    # Convertidor a listas
    all_genres = [genre for sublist in filtered_data if isinstance(sublist, str) for genre in safe_eval(sublist)]

    # Clasificacion
    most_common_genres = Counter(all_genres).most_common(5)
    top_genres = [genre[0] for genre in most_common_genres]
    
    return {
        "description": f"En el año {year} estos son los 5 géneros más ofrecidos de mayor a menor",
        "genres": top_genres
    }




@app.get("/juegos/")
def juegos(Año: str) -> dict:

    # Filtrado por el año
    filtered_games = [game['app_name'] for game in data if game['release_date'].startswith(Año)]
    
    return {
        "description": f"Juegos lanzados en el año {Año}",
        "games": filtered_games
    }



@app.get("/specs/")
def specs(Año: str) -> dict:
    
    # copia para no afectar al resto de las consultas.
    copia = data.copy()
    for game in copia:
        if game["specs"] and not isinstance(game["specs"], float):
            game["specs"] = literal_eval(game["specs"])
    
    # Filtrado por año
    games_of_year = [game for game in copia if Año in str(game['release_date'])]
    print(f"Juegos de {Año}: {games_of_year}")
  
    filtered_specs = [game['specs'] for game in games_of_year if isinstance(game['specs'], list)]
    all_specs = [spec for sublist in filtered_specs for spec in sublist]
    most_common_specs = Counter(all_specs).most_common(5)
    
    return {
        "description": f"Los 5 specs más comunes en juegos lanzados en el año {Año}",
        "specs": [spec[0] for spec in most_common_specs]
    }



@app.get("/earlyaccess/")
def earlyaccess(Año: str) -> dict:

    # Filtrado por año y "early acces" = True
    
    early_access_games = [game for game in data if game['release_date'].startswith(Año) and game['early_access'] == True]
    count_early_access = len(early_access_games)
    
    return {
        "description": f"Cantidad de juegos lanzados en {Año} con Early Access",
        "count": count_early_access
    }



@app.get("/sentiment/")
def sentiment(Año: str) -> dict:

    # Filtrado por año
    games_of_year = [game for game in data if game['release_date'].startswith(Año)]
    
    # Sentimiento, cuando no es nan.
    all_sentiments = [game['sentiment'] for game in games_of_year if game['sentiment'] is not None and game['sentiment'] != 'nan']
    
    # Filtrado solo por los siguietnes resultados:
    desired_sentiments = [
        "Mostly Positive", "Mixed", "Very Positive", "Overwhelmingly Positive", 
        "Very Negative", "Positive", "Mostly Negative", "Negative", "Overwhelmingly Negative"
    ]
    filtered_sentiments = [sentiment for sentiment in all_sentiments if sentiment in desired_sentiments]

    
    sentiment_counts = Counter(filtered_sentiments)
    sentiment_dict = dict(sentiment_counts)
    
    return {
        "description": f"Análisis de sentimientos de juegos lanzados en {Año}",
        "sentiments": sentiment_dict
    }



@app.get("/metascore/")
def metascore(Año: str) -> dict:

    # Filtrado por año
    all_games_of_year = [game for game in data if game['release_date'].startswith(Año)]
    
    # Filtrar vacios
    def has_valid_metascore(game):
        try:
            score = float(game.get('metascore'))
            return True
        except:
            return False

    games_with_metascore = [game for game in all_games_of_year if has_valid_metascore(game)]
    
    # Ordenado
    if games_with_metascore:
        sorted_games = sorted(games_with_metascore, key=lambda x: float(x['metascore']), reverse=True)
        top_5_games = sorted_games[:5]
        result = [{"title": game['app_name'], "metascore": game['metascore']} for game in top_5_games]

        return {
            "description": f"Top 5 juegos con el mayor metascore lanzados en {Año}",
            "games": result
        }
    else:
        return {"description": f"No hay juegos con metascore para el año {Año}"}


with open("columns.json", "r") as f:
    X_columns = json.load(f)




@app.post("/predict/")
def predict_price(genero: str, early_access: bool, metascore: float, sentiment: int, year: int):

    
    df = pd.DataFrame(np.zeros((1, len(X_columns))), columns=X_columns)

    
    df['early_access'] = int(early_access)
    df['metascore'] = metascore
    df['sentiment'] = sentiment
    df['year'] = year
    if genero in df.columns:
        df[genero] = 1

    
    predicted_price = model.predict(df)[0]
    
    with open("rmse.txt", "r") as file:
        stored_rmse = float(file.readline())

    
    return {
        "predicted_price": predicted_price,
        "RMSE": stored_rmse
    }





if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



