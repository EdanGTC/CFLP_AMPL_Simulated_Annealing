Código realizado por:
-Eduardo Gutiérrez Cerpa
-Matias Salinas Brito
# CFLP_AMPL_Simulated_Annealing

Indicaciones:
1) Tener previamente instalado python 3.8 o superior.
2) Instalar las librerías con los siguientes comandos (en caso de tenerlas saltarse este paso):
      -pip install pandas
      -pip install numpy
      -pip install amplpy
      -pip install matplotlib	
3) Se asume que el usuario tiene descargado una licencia original de AMPL (Archivo  ampl_mswin64).
4) El usuario deberá indicar la ruta de la carpeta ampl_mswin64 en el archivo path.txt


Ejecución del código:
1) Abrir la terminal y ubicarse en el directorio del código fuente (\lab2)
2) Ejecutar el comando “python lab2.py”, a este comando se le pueden incluir las siguientes opciones:

En caso de no ingresar una opción, los valores por defecto son:
--iterations = 500
--capacity = 7250
--plot = false 
--file = cap134.txt
--relaxed = false

Ejemplos de comandos:
“python lab2.py -i 100 -f cap131.txt -p” (Lee el archivo “cap131.txt”, realiza la heurística con 100 iteraciones y mostrará el gráfico correspondiente)
“python lab2.py --iterations 200 --file capa.txt --capacity 8000” (Lee el archivo “capa.txt”, realiza 200 iteraciones de la heurística y se le asigna a los centros una capacidad de 8000)
