# <h1 align=center> **PROYECTO INDIVIDUAL Nº1** </h1>

# <h1 align=center>**`Machine Learning Operations (MLOps)`**</h1>


## Primer paso

Comenze el trabajo convirtiendo el archivo json a csv, el cual es un entorno el cual me facilita la operacion y la visualizacion para entender de que se trataba la base de datos.

Esto lo realize en "ConversionCSV.ipynb"

## Segundo paso

Luego de eso comenze directamente a realizar las distintas funciones de la API.
Para eso, en vez de realizar una limpieza general y despues hacer las funciones, limpie especificamente lo que iba necesitando para cada funcion. Todo esto se ve reflejado en "main.py"

## Tercer paso

Habiendo finalizado las primeras 6 consultas de la API realize el EDA.
En primer lugar simplique los precios eliminando los outliders. Use los percentiles 25% 75%. Para poder entender mas adelante las distintas correlaciones con los precios y que se visualize mejor. 

Mire las distribuciones de las columnas que crei que eran mas significantes. Y luego mire la correlacion que tenia con el precio mediante distintos tipos de graficos. 

## Cuarto paso

Analisando el EDA, pude observar que las correlaciones con los precios no eran significativamente grandes, sino que mas bien eran debiles. Por lo tanto tome la decision de que para mi prediccion iba a utilizar mas de una. Utilice las que pedian en la consigna "Early Acces" y "Genred" y a eso le agregue "release_date (Year)", metascore y sentiment.

## Quinto paso

Luego de varias pruebas pude obtener un RMSE con el cual estube conforme. 4.8. De todas formas creo que no es suficientemente bajo para la distribucion de precios que tengo.

## Sexto paso

Introduje el modelo a la API y el RMSE por separado. linear_model.pkl y rmse.txt 
