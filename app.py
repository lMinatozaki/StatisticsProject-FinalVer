import json
import math
import webbrowser
#import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from scipy.stats import norm
import numpy as np
from tabulate import tabulate

#INSTALAR: Necesarias para el programa
#pip install pandas
#pip install matplotlib
#pip install scipy
#pip install numpy
#pip install tabulate

def readCsv(filename):
    #Lee el archivo CSV
    df = pd.read_csv(filename)
    return df
    
#Encontrar valor maximo y minimo
def findXmax(data):
    xmax = data[0]
    for number in data:
        if number > xmax:
            xmax = number
    return xmax

def findXmin(data):
    xmin = data[0]
    for number in data:
        if number < xmin:
            xmin = number
    return xmin

#Calcular el número de clases
def findK(data):
    n = len(data)
    k = 1 + 3.3 * math.log10(n)
    #print(f"k: {k}")
    return math.ceil(k)

#Calcular la amplitud
def findA(xmax, xmin, k):
    R = (xmax - xmin) + 1
    A = R / k
    #print(f"R: {R}")
    #print(f"A: {A}")
    return math.ceil(A)

#Construir intervalos de clase
def setIntervals(xmin, A, k):
    intervals = []
    limitI = xmin
    for _ in range(k):
        limitS = limitI + A
        intervals.append((limitI, limitS))
        limitI = limitS
    return intervals

#Contar
def countFrecuency(data, intervals):
    frecuency = [0] * len(intervals)
    for dataX in data:
        for i, (limitI, limitS) in enumerate(intervals):
            if limitI <= dataX < limitS:
                frecuency[i] += 1
                break
    return frecuency

#Calcular media aritmetica
def calcArithmeticMean(data, table):
    sumFixi = sum(item["fi.xi"] for item in table)
    n = len(data)
    arithmeticMean = sumFixi / n
    return arithmeticMean

#Calcular mediana
def calcMean(intervals, frecuency, n, A):
    acum = 0
    meanInterval = None
    
    for i, fi in enumerate(frecuency):
        acum += fi
        if acum >= n / 2:
            meanInterval = intervals[i]
            fiAnterior = acum - fi  #Frecuencia acumulada anterior
            Li = meanInterval[0]
            Med = Li + ((n / 2 - fiAnterior) / fi) * A
            return Med
    return None
    
#Calcular desviacion estandar y varianza
def calcVarianceAndStandarDeviation(table, n, x):
    sumFixi2 = sum(item["fi.xi^2"] for item in table)
    variance = (sumFixi2 / n) - math.pow(x, 2)
    
    standarDeviation = math.sqrt(variance)
    return variance, standarDeviation

#Calcular moda
def calcModal(intervals, frecuency, A):
    fiMax = max(frecuency)
    modals = []
    error = False

    for modalIndex, fi in enumerate(frecuency):
        if fi == fiMax:
            Li = intervals[modalIndex][0]
            
            fiAnterior = frecuency[modalIndex - 1] if modalIndex > 0 else 0
            #Si modalIndex > que 0, significa que existe una clase anterior, y por lo tanto asigna el valor
            #Si vale 0, significa que estamos en la primera clase y no hay clase anterior, por lo que se asigna 0
            
            fiSiguiente = frecuency[modalIndex + 1] if modalIndex < len(frecuency) - 1 else 0
            #Si modalIndex < que len(frecuency) - 1, significa que hay una clase que sigue y asigna el valor
            #Si vale lo mismo que len(frecuency) - 1, significa que estamos en la última clase y no hay clase siguiente, por lo que se asigna 0

            d1 = fi - fiAnterior
            d2 = fi - fiSiguiente
            
            if d1 == 0 and d2 == 0:
                modal = Li + A
                error = True
                errorString = "Excepción: clases cercanas a la moda tienen la misma frecuencia acumulada, por lo tanto, da una division entre cero.\nSe utiliza otra formula: (Li + A)"
                modals.append(modal)
            else:
                modal = Li + (d1 / (d1 + d2)) * A
                error = False
                errorString = ""
                modals.append(round(modal, 2))            
    return modals, error, errorString

#Calcular percentil
def calcPercentil(intervals, frecuency, n, A, k):
    position = (n * k) / 100
    acum = 0
    percInterval = None
    
    for i, fi in enumerate(frecuency):
        acum += fi
        if acum >= position:
            percInterval = intervals[i]
            fiAnterior = acum - fi  #Frecuencia acumulada anterior
            Li = percInterval[0]
            Pk = Li + ((position - fiAnterior) / fi) * A
            return Pk
    return None

#Calcular cuartil
def calcQuartil(data, intervals, frecuency, n, A):
    Q1 = calcPercentil(intervals, frecuency, n, A, 25)
    Q2 = calcPercentil(intervals, frecuency, n, A, 50)
    Q3 = calcPercentil(intervals, frecuency, n, A, 75)
    Q4 = np.max(data)
    return Q1, Q2, Q3, Q4

#Calcular deciles
def calcDeciles(intervals, frecuency, n, A):
    deciles = []
    for j in range(1, 10):
        decil = calcPercentil(intervals, frecuency, n, A, j * 10)
        deciles.append((j, decil))
    return deciles

#Calcular curtosis
def calcKurtosis(P75, P25, P90, P10):
    kurtosis = ((P75 - P25) / (P90 - P10)) * 0.5
    if kurtosis > 0:
        typeK = "leptocurtica"
    elif kurtosis == 0:
        typeK = "mesocurtica"
    else:
        typeK = "platicurtica"
    return kurtosis, typeK

#Calcular indice de asimetria
def calcAsymmetryIndex(arithmeticMean, mean, standarDeviation):
    if standarDeviation == 0:
        return float('nan')  #Nao nao division entre 0
    asymmetryIndex = 3 * (arithmeticMean - mean) / standarDeviation
    if asymmetryIndex == 0:
        typeA = "simétrica"
    elif asymmetryIndex < 0:
        typeA = "sesgo a la izquierda"
    else:
        typeA = "sesgo a la derecha"
    return asymmetryIndex, typeA

#Calcular coeficiente de variacion
def calcCoefficientOfVariation(standarDeviation, arithmeticMean):
    if arithmeticMean == 0:
        return float('nan')  #Nao nao division entre 0
    cv = standarDeviation / arithmeticMean
    return cv

#Calcular rango intercuartil
def calcInterquartileRange(Q1, Q3):
    Ri = Q3 - Q1
    return Ri

#Tabla de frecuencia
def createTable(intervals, frecuency, n):
    table = []
    fa = 0
    for i, (limitI, limitS) in enumerate(intervals):
        fi = frecuency[i]
        fa += fi
        fsr = fi / n  #Frecuencia simple relativa
        far = fa / n  #Frecuencia acumulada relativa
        xi = (limitI + limitS) / 2  #Punto medio
        fixi = fi * xi
        xi2 = xi*xi
        fixi2 = fi * xi2  #Fi*xi^2
        percFsr = fsr * 100
        percFar = far * 100
        classT = {
            "Intervalo": f"[{limitI}, {limitS})",
            "Fi": frecuency[i],
            "Fa": fa,
            "Fsr": round(fsr, 4),
            "Far": round(far, 4),
            "xi": round(xi, 2),
            "fi.xi": round(fixi, 2),
            "fi.xi^2": round(fixi2, 2),
            "Fsr%": percFsr,
            "Far%": percFar
        }
        table.append(classT)
    return table

def printTable(tableF):
    print('-' * 115)
    print(f"{'Intervalo':<20}{'Fi':<10}{'Fa':<10}{'Fsr':<10}{'Far':<10}"
          f"{'xi':<10}{'fi.xi':<15}{'fi.xi^2':<15}{'Fsr%':<10}{'Far%':<10}")
    print('-' * 115)
    for fila in tableF:
        interval = fila['Intervalo']
        fi = fila['Fi']
        fa = fila['Fa']
        fsr = fila['Fsr']
        far = fila['Far']
        xi = fila['xi']
        fixi = fila['fi.xi']
        fixi2 = fila['fi.xi^2']
        percFsr = fsr * 100
        percFar = far * 100
        print(f"{interval:<20}{fi:<10}{fa:<10}{fsr:<10}{far:<10}"
              f"{xi:<10}{fixi:<15}{fixi2:<15}{round(percFsr, 2):<10}{round(percFar, 2):<10}")
    print('-' * 115)

def loadTable(name):
    with open(name, 'r') as jsonX:
        tableF = json.load(jsonX)
    return tableF

def saveTable(table, name):
    with open(name, 'w') as jsonX:
        json.dump(table, jsonX)

def graphCuantitativos(df, variable):
    colors = {'Income': 'skyblue', 'HoursWk': 'salmon', 'Age': 'limegreen'}
    
    data = df[variable].dropna()
    color = colors.get(variable, 'grey')

    #Histograma
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=10, edgecolor='black', color=color, alpha=0.7)
    plt.title(f'Histograma de {variable}')
    plt.xlabel(variable)
    plt.ylabel('Frecuencia')
    plt.grid(True)
    plt.show()

    #Polígono de frecuencias
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=10, edgecolor='black', color=color, alpha=0.7, histtype='step', linewidth=2)
    plt.title(f'Polígono de Frecuencias de {variable}')
    plt.xlabel(variable)
    plt.ylabel('Frecuencia')
    plt.grid(True)
    plt.show()

    #Ojiva
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=10, cumulative=True, edgecolor='black', color=color, alpha=0.7, histtype='step', linewidth=2)
    plt.title(f'Ojiva de {variable}')
    plt.xlabel(variable)
    plt.ylabel('Frecuencia Acumulada')
    plt.grid(True)
    plt.show()

def graphCualitativos(df, variable):
    mappings = {
        'Sex': {0: 'Female', 1: 'Male'},
        'Married': {0: 'Not Married', 1: 'Married'},
        'USCitizen': {1: 'Citizen', 0: 'Non-Citizen'},
        'HealthInsurance': {1: 'Has Insurance', 0: 'No Insurance'},
        'Language': {1: 'English', 0: 'Other'}
    }
    
    if variable in mappings:
        data = df[variable].replace(mappings[variable]).dropna()
    else:
        data = df[variable].dropna()
    
    counts = data.value_counts()

    #Gráfico de barras
    plt.figure(figsize=(10, 6))
    counts.plot(kind='bar', color='skyblue')
    plt.title(f'Gráfico de Barras de {variable}')
    plt.xlabel(variable)
    plt.ylabel('Frecuencia')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()

    #Gráfico circular
    plt.figure(figsize=(8, 8))
    counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colormap='Pastel1')
    plt.title(f'Gráfico Circular de {variable}')
    plt.ylabel('')
    plt.show()

def showResults(df, variable):
    data = df[variable].dropna()
    n = len(data)
    xmax = findXmax(data)
    xmin = findXmin(data)
    k = findK(data)
    A = findA(xmax, xmin, k)
    intervals = setIntervals(xmin, A, k)
    frecuency = countFrecuency(data, intervals)
    tableF = createTable(intervals, frecuency, n)

    print('-' * 115)
    print(f"Tabla de frecuencias para {variable}:")
    printTable(tableF)

    arithmeticMean = calcArithmeticMean(data, tableF)
    mean = calcMean(intervals, frecuency, n, A)
    variance, standarDeviation = calcVarianceAndStandarDeviation(tableF, n, arithmeticMean)
    modal, error, errorString = calcModal(intervals, frecuency, A)
    P90 = calcPercentil(intervals, frecuency, n, A, 90)
    P75 = calcPercentil(intervals, frecuency, n, A, 75)
    P60 = calcPercentil(intervals, frecuency, n, A, 60)
    P50 = calcPercentil(intervals, frecuency, n, A, 50)
    P25 = calcPercentil(intervals, frecuency, n, A, 25)
    P10 = calcPercentil(intervals, frecuency, n, A, 10)
    Q1, Q2, Q3, Q4 = calcQuartil(data, intervals, frecuency, n, A)
    kurtosis, typeK = calcKurtosis(P75, P25, P90, P10)
    asymmetryIndex, typeA = calcAsymmetryIndex(arithmeticMean, mean, standarDeviation)
    cv = calcCoefficientOfVariation(standarDeviation, arithmeticMean)
    Ri = calcInterquartileRange(Q1, Q3)
    deciles = calcDeciles(intervals, frecuency, n, A)
    print("*" * 35)
    print(f"Análisis para {variable}:")
    print("*" * 35)
    print(f"Media Aritmética: {round(arithmeticMean, 2)}")
    print(f"Mediana: {round(mean, 2)}")
    print(f"Varianza: {round(variance, 2)}")
    print(f"Desviación estándar: {round(standarDeviation, 2)}")
    print("*" * 35)
    print(f"Modas: {modal}")
    if error:
        print(errorString)
    print("*" * 35)
    print(f"Curtosis: {round(kurtosis, 4)} - {typeK}")
    print(f"Índice de Asimetría: {round(asymmetryIndex, 4) if not math.isnan(asymmetryIndex) else 'No calculable'} - {typeA}")
    print(f"Coeficiente de variación: {round(cv*100, 4) if not math.isnan(asymmetryIndex) else 'No calculable'}%")
    print(f"Rango intercuartil: {round(Ri, 2)}")
    print(f"Cuartiles:\n Q1={round(Q1, 2)} \n Q2={round(Q2, 2)} \n Q3={round(Q3, 2)} \n Q4={round(Q4, 2)}")

def tableSexRace(df):
    sexMmapping = {0: 'F', 1: 'M'}
    raceMapping = {0: 'Asian', 1: 'Black', 2: 'White', 3: 'Other'}

    df['Sex'] = df['Sex'].replace(sexMmapping)
    df['Race'] = df['Race'].replace(raceMapping)

    # Contamos por raza y sexo
    table = df.groupby(['Race', 'Sex'], observed=True).size().unstack(fill_value=0)

    table['Total'] = table.sum(axis=1)
    totalRow = table.sum(axis=0)
    totalRow.name = 'Total'
    table = table._append(totalRow)

    filteredTable = table.drop(index='Total').drop(columns='Total')

    maxValue = filteredTable.stack().max()
    total = table.loc['Total', 'Total']
    maxGroups = filteredTable.stack()[filteredTable.stack() == maxValue].index.tolist()
    maxPorc = (maxValue / total) * 100

    print("*" * 35)
    print("Tabla de Raza y Sexo:")
    print(tabulate(table, headers='keys', tablefmt='grid'))
    print("\nLos grupos más grandes son:")
    for group in maxGroups:
        print(f"'{group}' con un {maxPorc:.2f}% del total.")

def tableHourAge(df):
    hourBins = [0, 19, 39, 59, float('inf')]
    hourLabels = ['0-19', '20-39', '40-59', 'Más horas']

    ageBins = [14, 24, 34, 44, 54, 64, 74, 84, 95]
    ageLabels = ['14-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75-84', '85-95']

    #Clasificamos los datos en intervalos
    df['Horas'] = pd.cut(df['HoursWk'], bins=hourBins, labels=hourLabels, right=False)
    df['Edad'] = pd.cut(df['Age'], bins=ageBins, labels=ageLabels, right=True)
    observed = True

    table = df.groupby(['Edad', 'Horas'], observed=True).size().unstack(fill_value=0)

    table['Total'] = table.sum(axis=1)
    totalRow = table.sum(axis=0)
    totalRow.name = 'Total'
    table = table._append(totalRow)

    filteredTable = table.drop(index='Total').drop(columns='Total')

    maxValue = filteredTable.stack().max()
    total = table.loc['Total', 'Total']
    maxGroups = filteredTable.stack()[filteredTable.stack() == maxValue].index.tolist()
    maxPorc = (maxValue / total) * 100

    print("*" * 35)
    print("Tabla de Horas y Edad:")
    print(tabulate(table, headers='keys', tablefmt='grid'))
    print("\nLos grupos más grandes son:")
    for group in maxGroups:
        print(f"'{group}' con un {maxPorc:.2f}% del total.")

def calcProb(df):
    meanIncome = df['Income'].mean()
    standarDeviationIncome = df['Income'].std()
    meanAge = df['Age'].mean()
    standarDeviationAge = df['Age'].std()
    meanHours = df['HoursWk'].mean()
    standarDeviationHours = df['HoursWk'].std()

    #P(X > 44)
    z1 = (44 - meanIncome) / standarDeviationIncome
    probP1 = 1 - norm.cdf(z1)
    print(f"Probabilidad de que el salario sea mayor que 44: {probP1 * 100:.4f}%")

    #P(47 < X < 49)
    z2_1 = (47 - meanIncome) / standarDeviationIncome
    z2_2 = (49 - meanIncome) / standarDeviationIncome
    probP2 = norm.cdf(z2_2) - norm.cdf(z2_1)
    print(f"Probabilidad de que el salario esté entre 47 y 49: {probP2 * 100:.4f}%")

    #P(D < 49)
    z3 = (49 - meanAge) / standarDeviationAge
    probP3 = norm.cdf(z3)
    print(f"Probabilidad de que la edad sea menor de 49 años: {probP3 * 100:.4f}%")

    #P(46 < D < 50)
    z4_1 = (46 - meanAge) / standarDeviationAge
    z4_2 = (50 - meanAge) / standarDeviationAge
    probP4 = norm.cdf(z4_2) - norm.cdf(z4_1)
    print(f"Probabilidad de que la edad esté entre 46 y 50 años: {probP4 * 100:.4f}%")

    #P(Y > 29.5)
    z5 = (29.5 - meanHours) / standarDeviationHours
    probP5 = 1 - norm.cdf(z5)
    print(f"Probabilidad de que las horas trabajadas sean mayor que 29.5: {probP5 * 100:.4f}%")

    #P(Y < 26.5)
    z6 = (26.5 - meanHours) / standarDeviationHours
    probP6 = norm.cdf(z6)
    print(f"Probabilidad de que las horas trabajadas sean menor que 26.5: {probP6 * 100:.4f}%")

def calcHipotesis(df):
    # ----- SALARIOS -----
    #Grupos aleatorios de salarios
    G1 = df['Income'].sample(n=500, random_state=1)
    G2 = df['Income'].sample(n=500, random_state=2)
    G3 = df['Income'].sample(n=400, random_state=3)

    meanG1, standardDeviationG1 = G1.mean(), G1.std()
    meanG2, standardDeviationG2 = G2.mean(), G2.std()
    meanG3, standardDeviationG3 = G3.mean(), G3.std()

    print("\n")
    print("*" * 35)
    print("Media y desviación estándar de cada grupo aleatorio de salario")
    print("*" * 35)
    print(f"Grupo 1 (G1): Media = {meanG1:.2f}, Desviación estándar = {standardDeviationG1:.2f}")
    print(f"Grupo 2 (G2): Media = {meanG2:.2f}, Desviación estándar = {standardDeviationG2:.2f}")
    print(f"Grupo 3 (G3): Media = {meanG3:.2f}, Desviación estándar = {standardDeviationG3:.2f}")

    #Hipotesis 1
    print("*" * 35)
    print("Prueba de hipotesis de G1 y G2")
    print("*" * 35)
    alpha95 = 0.05
    TStatG1andG2, PValueG1andG2 = stats.ttest_ind(G1, G2, equal_var=False)
    print(f"t-stat = {TStatG1andG2:.2f}, p-value = {PValueG1andG2:.4f}")
    if PValueG1andG2 < alpha95:
        print("Se rechaza H0, ya que las medias de G1 y G2 son significativamente diferentes al 95% del nivel de confianza.")
    else:
        print("No se puede rechazar H0, ya que no hay evidencia suficiente para afirmar que las medias de G1 y G2 son diferentes.")

    #Hipotesis 2
    print("*" * 35)
    print("Prueba de hipotesis de G2 y G3")
    print("*" * 35)
    alpha99 = 0.01
    TStatG2andG3, PValueG2andG3 = stats.ttest_ind(G2, G3, equal_var=False)
    print(f"t-stat = {TStatG2andG3:.2f}, p-value = {PValueG2andG3:.4f}")
    if PValueG2andG3 < alpha99:
        print("Se rechaza H0, ya que las medias de G2 y G3 son significativamente diferentes al 99% del nivel de confianza.")
    else:
        print("No se puede rechazar H0, ya que no hay evidencia suficiente para afirmar que las medias de G2 y G3 son diferentes.")
    print("*" * 35)

    # ----- EDADES -----
    #Grupos aleatorios de edades
    ageG1 = df['Age'].sample(n=300, random_state=4)
    ageG2 = df['Age'].sample(n=400, random_state=5)

    meanAgeG1, standardDeviationAgeG1 = ageG1.mean(), ageG1.std()
    meanAgeG2, standardDeviationAgeG2 = ageG2.mean(), ageG2.std()

    print("\n")
    print("*" * 35)
    print("Media y desviación estándar de cada grupo aleatorio de edad")
    print("*" * 35)
    print(f"G1: Media = {meanAgeG1:.2f}, Desviación estándar = {standardDeviationAgeG1:.2f}")
    print(f"G2: Media = {meanAgeG2:.2f}, Desviación estándar = {standardDeviationAgeG2:.2f}")

    #Hipotesis 3
    print("*" * 35)
    print("Prueba de hipotesis de G1 y G2")
    print("*" * 35)
    alpha90 = 0.10
    TStatAge, PValueAge = stats.ttest_ind(ageG1, ageG2, equal_var=False)
    print(f"t-stat = {TStatAge:.2f}, p-value = {PValueAge:.4f}")
    if PValueAge < alpha90:
        print("Se rechaza H0, ya que las medias de las edades son significativamente diferentes al 90% del nivel de confianza.")
    else:
        print("No se puede rechazar H0, ya que no hay evidencia suficiente para afirmar que las medias de las edades son diferentes.")
    print("*" * 35)
    print("\n")

def menu():
    print("*" * 35)
    print("*   STATISTICS PROJECT Ver. 2.0   *")
    print("*" * 35)
    print("*             OPTIONS             *")
    print("*" * 35)
    print("1. Ver resultados de Income")
    print("2. Ver resultados de HoursWk")
    print("3. Ver resultados de Age")
    print("4. Ver gráficos de Income")
    print("5. Ver gráficos de HoursWk")
    print("6. Ver gráficos de Age")
    print("7. Ver gráficos de Sex")
    print("8. Ver gráficos de Married")
    print("9. Ver gráficos de Race")
    print("10. Ver gráficos de USCitizen")
    print("11. Ver gráficos de HealthInsurance")
    print("12. Ver gráficos de Language")
    print("13. Ver tabla Raza/Sexo")
    print("14. Ver tabla Horas/Edad")
    print("15. Ver probabilidades")
    print("16. Ver hipotesis")
    print("17. Salir")
    print("*" * 35)

def main():
    filename = 'Datos proyecto 2024.csv'
    df = readCsv(filename)

    while True:
        menu()
        opcion = input("Seleccione una opción: ")

        if opcion == '1':
            showResults(df, 'Income')
        elif opcion == '2':
            showResults(df, 'HoursWk')
        elif opcion == '3':
            showResults(df, 'Age')
        elif opcion == '4':
            graphCuantitativos(df, 'Income')
        elif opcion == '5':
            graphCuantitativos(df, 'HoursWk')
        elif opcion == '6':
            graphCuantitativos(df, 'Age')
        elif opcion == '7':
            graphCualitativos(df, 'Sex')
        elif opcion == '8':
            graphCualitativos(df, 'Married')
        elif opcion == '9':
            graphCualitativos(df, 'Race')
        elif opcion == '10':
            graphCualitativos(df, 'USCitizen')
        elif opcion == '11':
            graphCualitativos(df, 'HealthInsurance')
        elif opcion == '12':
            graphCualitativos(df, 'Language')
        elif opcion == '13':
            tableSexRace(df)
        elif opcion == '14':
            tableHourAge(df)
        elif opcion == '15':
            calcProb(df)
        elif opcion == '16':
            calcHipotesis(df)
        elif opcion == '17':
            print("Saliendo del programa...")
            break
        else:
            print("Opción no válida. Intente de nuevo.")

if __name__ == "__main__":
    main()