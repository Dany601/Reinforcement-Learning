#Proyecto de Aprendizaje por Refuerzo - Navegación de un Laberinto con Q-Learning
Este proyecto aplica Q-Learning, un algoritmo de aprendizaje por refuerzo, para entrenar a un agente que navega en un laberinto de 100x100 hasta alcanzar un objetivo definido. Se utiliza Python y bibliotecas como NumPy para implementar y probar el modelo.

##Objetivo
Entrenar un agente inteligente para que resuelva un laberinto definido, aplicando conceptos de aprendizaje por refuerzo y visualizando el camino óptimo hacia el objetivo.

##Contenido del Proyecto
###1. Configuración del Laberinto
Se diseñó un laberinto de tamaño 100x100 representado mediante una matriz de NumPy, donde cada celda tiene el siguiente significado:

-**0**: Camino libre.
-**1**: Obstáculo o pared.
-**2**: Celda objetivo.
El laberinto se generó con obstáculos aleatorios, asegurando que exista al menos un camino hacia el objetivo.

###2. Parámetros del Modelo
El agente puede realizar cuatro acciones:

-**Arriba**: Movimiento hacia la celda superior.
-**Abajo**: Movimiento hacia la celda inferior.
-**Izquierda**: Movimiento hacia la celda izquierda.
-**Derecha**: Movimiento hacia la celda derecha.
Los movimientos son validados para garantizar que el agente no salga de los límites del laberinto ni atraviese obstáculos.
Parámetros utilizados para el algoritmo de Q-Learning:

-**Alpha** (tasa de aprendizaje): 0.1
-**Gamma** (factor de descuento): 0.9
-**Epsilon** (exploración-explotación): 0.2
###3. Implementación en Python
El proyecto se desarrolló utilizando Python en Google Colab, empleando las siguientes herramientas:

-**NumPy**: Para la creación y manipulación del laberinto.
-**Matplotlib**: Para la visualización gráfica del recorrido del agente.
-**Q-Learning**: Implementado desde cero con una tabla Q para almacenar las recompensas aprendidas.
####Pasos del Algoritmo
1.Inicialización del laberinto y los valores Q.
2.Entrenamiento del agente durante 1000 episodios, ajustando sus políticas según las recompensas obtenidas.
3.Evaluación del agente en el laberinto entrenado, mostrando el recorrido desde el punto de inicio hasta la meta.
###4. Resultados y Visualización
1.Camino aprendido: El agente encontró una trayectoria óptima para alcanzar la meta desde un punto inicial predeterminado.
2.Visualización gráfica: Se generó un gráfico del laberinto mostrando:
-**Las celdas visitadas**.
-**El camino final seguido por el agente**.
-**El objetivo alcanzado**.
###5. Evaluación del Modelo
El desempeño del agente fue evaluado midiendo:

-**Número de pasos requeridos para alcanzar la meta**.
-**Tasa de éxito en diferentes configuraciones del laberinto**.
-**Convergencia del algoritmo analizando la evolución de la tabla Q**.
## Enlaces de Referencia
-**Colab Notebook**: Accede al proyecto en Google Colab

## Instrucciones para Ejecutar el Proyecto
1. Clonar este repositorio:
        git clone https://github.com/tu_usuario/tu_repositorio.git
2. Abrir el archivo en Google Colab o cualquier entorno local con Python.
3. Instalar las dependencias necesarias
        pip install numpy matplotlib
4. Ejecutar el script principal y observar los resultados.
